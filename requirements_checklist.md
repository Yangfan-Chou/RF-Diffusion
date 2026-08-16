# RF-Diffusion 项目要求检查清单

## 项目信息
- **论文**: RF-Diffusion: Radio Signal Generation via Time-Frequency Diffusion
- **会议**: ACM MobiCom 2024
- **项目名称**: RF-Diffusion：面向无线信号生成的时频扩散模型复现与低算力推理评估

## 五个核心环节

### 1. 阅读文献 ✓
- [ ] 阅读RF-Diffusion论文完整内容
- [ ] 阅读论文附录
- [ ] 阅读官方README
- [ ] 阅读核心代码文件
- [ ] 调研相关工作(6-10篇)

### 2. 提炼思路 ✓
- [ ] 理解研究背景
- [ ] 识别核心科学问题
- [ ] 梳理RF数据与图像数据的差异
- [ ] 分析时间序列特性
- [ ] 分析频域特性
- [ ] 分析复数域特性
- [ ] 理解Time-Frequency Diffusion理论
- [ ] 分析Hierarchical Diffusion Transformer结构

### 3. 复现算法 ✓
- [ ] Level 1: 官方结果可执行复现
  - [ ] 安装官方环境
  - [ ] 下载官方数据和预训练模型
  - [ ] 运行官方绘图脚本
  - [ ] 运行Wi-Fi任务 (python inference.py --task_id 0)
  - [ ] 条件允许时运行5G任务 (python inference.py --task_id 2)
  - [ ] 保存全部终端日志
  - [ ] 验证输出包含 Average SSIM, FID, Average SNR
- [ ] Level 2: 标准化复现框架
  - [ ] argparse或YAML配置
  - [ ] 自动设备选择
  - [ ] CPU/GPU兼容
  - [ ] 路径跨平台兼容
  - [ ] 固定随机种子
  - [ ] 日志记录
  - [ ] 实验配置保存
  - [ ] 推理时间统计
  - [ ] 峰值显存统计
  - [ ] 峰值内存统计
  - [ ] 异常处理
  - [ ] 单样本快速测试
  - [ ] 数据和模型完整性检查
- [ ] Level 3: 缩小规模训练
  - [ ] 使用公开Wi-Fi数据子集
  - [ ] 减少训练样本数
  - [ ] 减少epoch
  - [ ] 选择较小的hidden dimension或block数量
  - [ ] 保留Time-Frequency Diffusion关键逻辑
  - [ ] 训练过程保存loss曲线、checkpoint、配置文件
  - [ ] 明确标注为small-scale reproduction

### 4. 总结凝练 ✓
- [ ] 独立思考撰写critical_analysis.md
- [ ] 分析RF-Diffusion相对DDPM、DCGAN、CVAE的优势
- [ ] 分析时间域加噪和频域模糊设计的合理性
- [ ] 分析复数域建模的重要性
- [ ] 分析Hierarchical Diffusion Transformer的必要性
- [ ] 分析跨无线任务通用性
- [ ] 分析FID用于RF频谱质量评价的局限
- [ ] 分析SSIM是否充分反映RF物理特征
- [ ] 分析预训练模型复现与从头训练复现的差异
- [ ] 分析扩散模型推理速度较慢的问题
- [ ] 分析边缘设备和工业现场部署问题
- [ ] 分析数据分布变化、场景变化和设备差异的影响
- [ ] 提出至少三个有逻辑依据的改进方向

### 5. 汇总报告 ✓
- [ ] LaTeX报告 (1500-2200中文字符)
- [ ] 摘要 (150-250字)
- [ ] 关键词 (4-6个)
- [ ] 至少3个规范编号公式
- [ ] 至少2张图
- [ ] 至少2个表格
- [ ] RF信号复数形式展示
- [ ] BibTeX管理参考文献

## 环境审计
- [ ] Windows/Linux/WSL检测
- [ ] Python版本记录
- [ ] NVIDIA GPU型号记录
- [ ] CUDA版本记录
- [ ] 可用显存记录
- [ ] CPU型号记录
- [ ] 系统内存记录
- [ ] 剩余磁盘空间记录
- [ ] Git/Conda/LaTeX安装检测

## 目录结构
- [ ] README.md
- [ ] requirements_checklist.md
- [ ] AI_USAGE.md
- [ ] LICENSE_NOTICE.md
- [ ] environment/
- [ ] configs/
- [ ] data/
- [ ] docs/
- [ ] notebooks/
- [ ] results/
- [ ] figures/
- [ ] scripts/
- [ ] src/
- [ ] tests/
- [ ] upstream/RF-Diffusion/
- [ ] report/

## 数据和模型
- [ ] 官方仓库克隆至upstream/RF-Diffusion
- [ ] 记录上游commit hash
- [ ] 下载脚本编写
- [ ] 文件完整性校验
- [ ] 数据和模型不提交到Git

## 性能评测指标
- [ ] SSIM
- [ ] FID
- [ ] SNR (5G实验)
- [ ] 单样本推理时间
- [ ] 总推理时间
- [ ] 峰值GPU显存或CPU内存
- [ ] 模型参数量
- [ ] 模型文件大小

## 扩展实验主题
"计算预算约束下RF-Diffusion生成质量与推理效率的权衡"
- [ ] 不同模型规模对比 (16/32 blocks, 128/256 hidden dims)
- [ ] 不同推理采样策略对比
- [ ] 不同采样步数对比
- [ ] FP32与AMP/FP16对比
- [ ] 不同测试样本数量下的耗时变化
- [ ] CPU与GPU对比
- [ ] 原始时域与频谱域可视化对比

## 图表要求
- [ ] 质量—推理时间折线图或散点图
- [ ] 模型规模—峰值内存柱状图
- [ ] 真实信号与生成信号频谱对比图
- [ ] 论文结果与复现结果对比表
- [ ] 至少一个消融或敏感性分析
- [ ] 使用matplotlib
- [ ] PDF和PNG双格式保存
- [ ] 分辨率不低于300 DPI
- [ ] 学术论文风格

## 科研诚信
- [ ] 不编造实验结果
- [ ] 所有数值可追溯到results目录
- [ ] 论文结果标注"Paper reported"
- [ ] 本项目结果标注"Reproduced"
- [ ] 失败实验写明原因
- [ ] 不直接复制论文段落
- [ ] 论文公式依据原文核对
- [ ] 修改过的官方代码记录
- [ ] 固定随机种子
- [ ] AI_USAGE.md说明

## 代码质量
- [ ] Python类型注解
- [ ] 关键函数docstring
- [ ] 使用logging
- [ ] 配置和代码分离
- [ ] 异常处理
- [ ] 单元测试
- [ ] ruff/flake8检查
- [ ] pytest执行测试
- [ ] Windows与Linux路径兼容
- [ ] PATCH_NOTES.md记录修改
- [ ] YAML配置文件
- [ ] 唯一run_id
- [ ] 实验输出不相互覆盖

## LaTeX报告要求
- [ ] XeLaTeX编译
- [ ] ctexart文档类
- [ ] 标准学术论文结构
- [ ] 三个规范编号公式
- [ ] RF信号复数形式: x = x^{Re} + jx^{Im}
- [ ] 表1: 相关生成方法比较
- [ ] 表2: 论文报告与复现结果比较
- [ ] 图1: RF-Diffusion总体流程图
- [ ] 图2: 真实与生成信号频谱对比
- [ ] 图3(可选): 质量—效率权衡图
- [ ] booktabs三线表
- [ ] BibTeX参考文献

## 一键复现
- [ ] scripts/reproduce_all.sh (Linux/WSL)
- [ ] scripts/reproduce_all.ps1 (Windows)
- [ ] notebooks/RF_Diffusion_Reproduction.ipynb (Colab)
- [ ] 按顺序执行: 环境检查→数据检查→模型检查→smoke test→正式推理→指标计算→图表生成→结果汇总→LaTeX编译

## 最终验收
- [ ] 运行完整自检
- [ ] 生成FINAL_AUDIT.md
- [ ] 逐项检查所有要求
- [ ] 诚实说明与论文差异
