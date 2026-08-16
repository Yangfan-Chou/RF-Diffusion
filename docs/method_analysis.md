# RF-Diffusion 核心方法分析

## 1. 总体框架

RF-Diffusion 是一个时频联合扩散（Time-Frequency Diffusion）框架，核心思想是将原始DDPM的"时域加噪"扩展为"时域加噪 + 频域模糊"的联合退化过程。整个流程由两部分组成：

1. **退化过程 D(·, t)**：把复数RF信号 x_0 变成退化的 x_t，其中频域逐级平滑且叠加高斯噪声。
2. **恢复过程（去噪网络）**：训练神经网络 (Hierarchical Diffusion Transformer, tfdiff) 学习从 x_t 预测原始 x_0。

## 2. 退化过程（Degradation Process）

论文Section 5给出退化过程定义：

### 2.1 频域模糊核 G_t

每个时间步 t 应用一个方差为 \sigma_t^2 的高斯核 G_t 在频域上对信号做卷积。

高斯核生成（在代码中通过 `get_kernel` 实现）：
$$
G_t[n] = \frac{N}{\sum_m G_t[m]} \cdot \frac{1}{\sqrt{2\pi \sigma_t^2}} \exp\left(-\frac{(n - N/2)^2}{2 \sigma_t^2}\right)
$$

论文中的等价噪声权重 \bar{w}_t 由累积频域模糊得到：
$$
\bar{w}_t = \frac{\bar{G_t}}{G_t} \cdot \sqrt{\beta_t \prod_{s<t} (1-\beta_s)}
$$

### 2.2 时域加噪

类似于DDPM：
$$
\epsilon_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

### 2.3 联合退化

$$
\mathbf{x}_t = D(\mathbf{x}_0, t) = \bar{G_t} * \mathbf{x}_0 + \bar{w}_t \cdot \epsilon_t
$$

论文的核心洞察：在频域上加高斯模糊等价于在时域上对信号乘以一个低通滤波后的系数，同时在时域上叠加高斯噪声。这个退化过程对应RF信号在传播过程中经过的多径叠加和噪声污染。

## 3. 训练目标

论文Section 5.3训练目标是最小化预测误差：

$$
\mathcal{L}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[ \| \hat{\mathbf{x}}_0(\mathbf{x}_t, t, \mathbf{c}) - \mathbf{x}_0 \|_2^2 \right]
$$

即训练神经网络直接预测 x_0，而非噪声 \epsilon 或 \mu。这与DDPM的noise-prediction不同，但作者实验发现 x_0-prediction对RF信号更稳定。

## 4. 推理采样

代码中实现了四种采样方式：
1. **`sampling`**：标准DDPM风格迭代去噪，T步。
2. **`robust_sampling`**：使用相邻两步差分更新（DDIM-like），更稳定。
3. **`fast_sampling`**：单步去噪（仅取 T-1 步预测），速度最快。
4. **`native_sampling`**：使用真实 x_0 作为参考加噪到 T-1 步然后去噪，用于评估目的。

论文主要使用 `native_sampling` 评估（匹配论文Section 6.2）。

## 5. 网络结构 t fdiff（Hierarchical Diffusion Transformer）

### 5.1 Wi-Fi任务（task_id=0）

```
输入 x: [B, 512, 90, 1, 2] (B=batch, 512=sample_rate, 90=input_dim, 1=extra_dim, 2=复数)
↓
PositionEmbedding (ComplexLinear → 复数位置编码)
↓
DiffusionEmbedding (时间步嵌入，复数傅立叶特征)
↓
MLPConditionEmbedding (条件信号嵌入)
↓
DiA Block × 32:
  ├─ adaLN modulation (ComplexSiLU + ComplexLinear)
  ├─ ComplexMultiHeadAttention
  ├─ Complex MLP
  └─ Residual + adaLN gate
↓
FinalLayer (ComplexLinear → 输出)
↓
输出: [B, 512, 90, 1, 2]
```

### 5.2 MIMO/5G任务（task_id=2）

为了处理空间-时频多维结构，采用双层Hierarchical结构：
- Spatial block (16 blocks)：处理每个空间切片。
- Time-Frequency block (16 blocks)：跨时间和频率维度建模。

## 6. 复数算子实现

`complex/complex_module.py` 实现了完整的复数深度学习算子：

| 算子 | 用途 |
|------|------|
| ComplexLinear | 复数全连接 |
| ComplexSiLU/GELU/ReLU | 复数激活 |
| NaiveComplexLayerNorm | 复数层归一化 |
| NaiveComplexBatchNorm3d | 复数批归一化 |
| ComplexMultiHeadAttention | 复数多头注意力 |
| ComplexDotProductAttention | 复数点积注意力 |
| ComplexConv3d | 复数3D卷积 |
| ComplexResidual3d | 复数残差块 |
| complex_mul/complex_bmm | 复数乘法/批矩阵乘 |

复数乘法的实现遵循复数域基本规则：
$$
(a+bi)(c+di) = (ac - bd) + (ad + bc)i
$$

代码中通过 `apply_complex(F_r, F_i, X)` 统一封装：
```python
def apply_complex(F_r, F_i, X):
    X_r, X_i = [...对X按最后一维拆分...]
    return stack(F_r(X_r) - F_i(X_i), F_r(X_i) + F_i(X_r), dim=-1)
```

## 7. 关键设计选择

### 7.1 为什么需要复数算子？
- RF信号天然是复数（I/Q双通道）。
- 实部虚部之间存在物理约束（包络、相位），分别建模会丢失该约束。
- 复数Linear/Attention可以保持包络和相位的内在关系。

### 7.2 为什么需要频域模糊？
- RF信号传播过程中多径叠加表现为频域选择性衰落。
- 论文中通过频域模糊模拟这种物理过程，比单纯时域加噪更贴近真实。
- 实验显示去除频域模糊后FID和SSIM都下降（论文Fig.9）。

### 7.3 为什么使用 Hierarchical 结构？
- Wi-Fi信号维度较高（512×90），单层Transformer计算开销大。
- Spatial-TF 双层结构可以分别处理空间局部和时间-频率全局依赖。
- 减少参数数量同时保持建模能力。

### 7.4 adaLN（adaptive LayerNorm）调制
- 类似 DiT (Peebles & Xie, ICCV 2023) 的设计。
- 时间步和条件信息通过调制 shift/scale 注入每个Block。
- 比 cross-attention 更高效且效果相当。

## 8. 与DDPM的关键区别

| 特性 | DDPM | RF-Diffusion |
|------|------|--------------|
| 退化域 | 仅时域 | 时域 + 频域 |
| 信号域 | 实数图像 | 复数RF信号 |
| 网络 | UNet | Hierarchical Diffusion Transformer |
| 训练目标 | 预测 \epsilon 或 \mu | 直接预测 x_0 |
| 任务 | 单一图像生成 | Wi-Fi/FMCW/5G多任务 |
| 复数算子 | 不需要 | 必备 |

## 9. 关键公式总结

1. 前向退化：
$$
\mathbf{x}_t = \bar{G_t} * \mathbf{x}_0 + \bar{w}_t \cdot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

2. 训练目标：
$$
\mathcal{L} = \| \hat{\mathbf{x}}_0 - \mathbf{x}_0 \|_2^2
$$

3. SSIM：
$$
\mathrm{SSIM}(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

4. SNR (dB)：
$$
\mathrm{SNR} = 10 \log_{10} \frac{\sum \| \mathbf{x}_{\text{truth}} \|^2}{\sum \| \mathbf{x}_{\text{pred}} - \mathbf{x}_{\text{truth}} \|^2}
$$

## 10. 代码模块对应关系

| 论文概念 | 代码实现 |
|---------|---------|
| 时频扩散 | `tfdiff/diffusion.py::SignalDiffusion` |
| 频域模糊核 | `SignalDiffusion.get_kernel()` |
| 高斯扩散 | `tfdiff/diffusion.py::GaussianDiffusion` |
| 时间步嵌入 | `tfdiff/wifi_model.py::DiffusionEmbedding` |
| 条件嵌入 | `tfdiff/wifi_model.py::MLPConditionEmbedding` |
| 位置编码 | `tfdiff/wifi_model.py::PositionEmbedding` |
| DiA Block | `tfdiff/wifi_model.py::DiA` |
| 复数算子 | `complex/complex_module.py` |
| 数据加载 | `tfdiff/dataset.py` |
| 训练 | `tfdiff/learner.py::tfdiffLearner` |
| 推理 | `inference.py` |