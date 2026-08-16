# RF-Diffusion 论文笔记

## 1. 研究背景
无线感知与通信的发展使研究人员需要大量多样化、高质量的RF信号数据用于训练下游识别与估计模型。但是真实RF信号数据采集成本高、场景固定、标签稀缺，限制了数据驱动的深度学习模型在无线感知和通信领域的应用。RF-Diffusion (MobiCom 2024) 提出了一种统一的复数时频扩散模型，可以从条件信号中恢复/补全原始复数RF信号，跨越Wi-Fi感知、FMCW雷达、5G信道估计等多种任务。

## 2. 核心科学问题
RF信号是复数时频序列，而现有图像扩散模型（DDPM/DCGAN/CVAE等）仅建模实数域图像，不能直接生成复数RF信号。RF-Diffusion的核心科学问题是：
> 如何设计一个既能建模RF复数信号，又能同时建模时域加噪和频域模糊（联合时-频退化）的扩散模型？

## 3. RF数据与图像数据的差异
1. **数据形式**：图像是二维实数像素矩阵，RF数据是复数序列（I/Q双通道）。
2. **物理意义**：图像像素是空间位置的颜色/灰度值，RF数据是载波调制后的时变复振幅。
3. **特征分布**：图像通常在[0,255]或[0,1]之间；RF复数信号在I/Q平面内呈圆对称分布，振幅服从瑞利分布。
4. **时频局部性**：图像局部相关，RF信号同时具备时间相关性和频率相关性（频谱结构）。
5. **频域稀疏性**：Wi-Fi/雷达信号在频域呈现稀疏子载波结构。

## 4. 时间序列特性
- 时域信号具有短时平稳性（几ms到几十ms）。
- 多径与多普勒效应引入时变包络。
- CSI（信道状态信息）的幅值和相位随时间演化。
- 时间相关性意味着时域加噪过程应在短窗口内进行局部平滑化操作。

## 5. 频域特性
- Wi-Fi信号在OFDM子载波上呈现频域稀疏结构。
- FMCW雷达的Range-Doppler Map（RDM）是稀疏二维频谱。
- 5G信号在频域具有导频符号和参考符号。
- RF-Diffusion通过频域模糊（Gaussian Blur on spectrogram）模拟频域信息丢失。

## 6. 复数域特性
- RF信号可表示为 x = x^{Re} + jx^{Im}。
- 复数乘法的实部和虚部必须协同计算。
- RF-Diffusion实现了ComplexLinear、ComplexSiLU、ComplexMultiHeadAttention等完整的复数算子。
- 时间步嵌入、条件嵌入、位置嵌入均使用Complex域。

## 7. Time-Frequency Diffusion理论
前向退化过程（D(·, t)）同时包含两个分量：
1. **时域加噪**：标准DDPM风格的高斯噪声注入 \epsilon_t。
2. **频域模糊**：在频域用方差为 \sigma_t^2 的高斯核 G_t 进行卷积。

数学表达（论文第5节）：
- x_t = D(x_0, t) = \bar{G_t} * x_0 + \bar{w_t} * \epsilon，\epsilon \sim N(0, I)
- \bar{G_t} = G_1 * G_2 * ... * G_t
- \bar{w_t} = \sqrt{1 - \alpha_t} 等价噪声权重

与DDPM相比，关键差异是：RF-Diffusion的退化过程不是单纯的时域加噪，而是时-频联合退化。论文指出 RF-Diffusion 退化过程对应"对原始信号的频域低通滤波加上同分布高斯噪声"，这与RF数据的物理特性更匹配。

## 8. Hierarchical Diffusion Transformer结构
RF-Diffusion的核心网络 t fdiff（Time-Frequency Diffusion Transformer）：
- 输入复数序列 x ∈ C^{B×N×S×2}
- PositionEmbedding：将输入投影到hidden_dim复数空间，并叠加复数正弦位置编码。
- DiffusionEmbedding：时间步嵌入使用复数傅立叶特征。
- MLPConditionEmbedding：条件信号通过MLP投影。
- DiA Block：Diffusion Attention Block，每个块包含ComplexMultiHeadAttention、Complex MLP和adaptive LayerNorm (adaLN)。
- 输出层 FinalLayer 使用adaLN将隐藏特征映射回原始维度。

对于不同任务（Wi-Fi、FMCW、MIMO/5G），论文使用不同的网络配置。Wi-Fi任务使用32个块，hidden_dim=128；FMCW任务使用32个块，hidden_dim=256；MIMO任务使用16+16的空间-时频双层结构。

## 9. 复数算子设计
RF-Diffusion在 `complex/complex_module.py` 中实现了完整的复数深度学习算子：
- ComplexLinear：分别对实部和虚部进行Linear变换后按复数乘法规则组合。
- ComplexSiLU / ComplexGELU / ComplexReLU：分别对实部虚部应用激活函数。
- ComplexMultiHeadAttention：在复数域计算 QKV 并做复数点积。
- ComplexDotProductAttention：使用 complex_softmax 和 complex_bmm。
- NaiveComplexLayerNorm / NaiveComplexBatchNorm3d：复数归一化。
- complex_mul / complex_bmm：基本复数矩阵运算。
- apply_complex(F_r, F_i, X)：复数算子的统一封装，遵循 (a+bi)(c+di) = (ac-bd) + (ad+bc)i。

## 10. 三个实验任务

### 10.1 Wi-Fi信号生成（task_id=0）
- 数据来源：Widar3.0数据集，CSI数据。
- 输入维度：sample_rate=512, input_dim=90, extra_dim=[90], cond_dim=6。
- 模型配置：b32-256-100s（32 blocks, 256 embed_dim, 100 diffusion steps）。
- 输出指标：Average SSIM, FID。
- 数据形状：(B, N, S, A, 2) = (B, 512, 90, 1, 2) 实部虚部。

### 10.2 FMCW雷达信号生成（task_id=1）
- 数据来源：mmWave FMCW Radar数据集。
- 输出指标：Average SSIM, FID。
- 模型配置：b32-256-100s（32 blocks, hidden_dim=256）。

### 10.3 5G FDD信道估计（task_id=2, MIMO）
- 数据来源：Argos数据集，下行链路信道估计。
- 输入维度：sample_rate=14, extra_dim=[26, 96], cond_dim=[26, 96]。
- 模型配置：b32-256-200s，使用 Spatial+TimeFrequency 双层结构（16+16 blocks）。
- 输出指标：Average SNR (dB)。

### 10.4 EEG信号去噪（task_id=3，未在主论文中）
- 模型配置：b32-256-200s，16 blocks。
- 输出指标：Average SNR (dB)。
- 本项目不重点复现。

## 11. 论文贡献
1. 提出Time-Frequency Diffusion，第一个将时域加噪和频域模糊同时融入扩散过程的RF信号生成框架。
2. 设计Hierarchical Diffusion Transformer（tfdiff）网络结构，专为复数RF信号建模。
3. 实现完整的Complex-valued深度学习算子库。
4. 在Wi-Fi、FMCW、5G三大任务上同时验证生成质量超越DDPM/DCGAN/CVAE基线。
5. 验证合成数据可作为数据增强手段提升下游感知性能。
6. 验证5G FDD下行信道估计任务上RF-Diffusion超越state-of-the-art方法。

## 12. 论文局限
1. 推理速度较慢，需要多次前向传播（采样步数等于max_step=100/200）。
2. 训练数据集分布有限，跨设备、跨场景泛化能力未充分验证。
3. FID指标使用预训练InceptionV3，未针对RF信号进行领域适配。
4. SSIM主要反映像素级结构相似性，未直接反映RF物理特征（如多径、时延）。
5. 论文使用 `corr = 1.9`（Wi-Fi）和 `corr = 0.9`（FMCW）对FID进行经验校正，缺乏理论依据。
6. 没有显式的物理约束损失（如恒包络、导频位置等）。
7. 模型规模较大（32 blocks），难以直接部署到边缘设备。

## 13. 对工业物联网、无线感知和边缘智能的意义
- **数据增强**：缓解工业场景RF数据稀缺问题，降低数据采集成本。
- **隐私保护**：可生成合成数据替代敏感真实信号。
- **跨设备/跨场景泛化**：通过条件生成实现模型在不同设备配置下的快速适配。
- **边缘部署挑战**：扩散模型推理速度慢，需要模型蒸馏或加速采样。
- **工业数字孪生**：可生成多场景、多设备的RF信号用于仿真测试。
- **6G通信**：为通感一体（ISAC）提供信号仿真基础。

## 公式核对（论文Section 5）

论文中前向过程与论文原文一致（参考文献：Chi et al., MobiCom 2024, Section 5）。

x_0 的复数表示：
$$
\mathbf{x}_0 = x_0^{Re} + j x_0^{Im}
$$

退化过程：
$$
\mathbf{x}_t = D(\mathbf{x}_0, t) = \bar{G_t} * \mathbf{x}_0 + \bar{w}_t * \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

$$
\bar{G_t} = G_1 * G_2 * \cdots * G_t
$$

训练目标：神经网络预测 $\hat{x}_0$（论文 Section 5.3）：

$$
\mathcal{L} = \| \hat{\mathbf{x}}_0 - \mathbf{x}_0 \|_2^2
$$

SSIM定义（Wang et al., 2004）：
$$
\mathrm{SSIM}(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

FID定义（Heusel et al., 2017）：基于InceptionV3池化特征的Fréchet距离。