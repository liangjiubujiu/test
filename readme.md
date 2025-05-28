# ctooth 牙齿3D分割
## 项目简介
在我们的工作之前，用作研究用途的高质量3D牙齿CBCT数据几乎没有开源，这一情况极大限制了当时的牙齿3D分割算法的研究，为解决这一痛点，我们花费了很多精力收集和标注牙齿CBCT数据。后续我们的数据随着2023年的MICCAI挑战赛进行了开源，以证明我们数据开源工作是真实且有贡献的。同时，为了方便后续工作者能将ctooth数据用于自动化牙齿分割任务以辅助牙医诊疗场景，我们不得不复现在其他领域相对成熟的开源3D分割方法作为对照方法，并统计了它们在ctooth上的分割效果。实验结果表明，这些对照方法在ctooth上表现不足，主要原因一方面是一些方法是为其他3D分割任务或通用3D分割任务而生，缺乏针对牙齿3D小目标的架构设计，比如DenseVoxelNet和3D HighResNet均是被提出用来在MR中分割心脏。另一方面，一些模型参数量过大在我们的数据集上可能存在过拟合，例如，3D Unet和VNet模型参数量较大，深度较深往往在5层左右，并且层间包含的3D Maxpooling操作易造成小的牙齿局部特征损失。


针对上述问题，我们提出了一个基于注意力的U型轻量框架作为分割基准，该框架允许研究者替换不同的attention模块，从而观察不同实验条件下的分割结果。目前，我们已对当时的ctooth工作进行了整理，补充了数据预处理、模型训练、推理的代码，现开源供对算法部分感兴趣的工作者参考。

近期，有人在未与我们事先沟通的情况下，对ctooth在当时已开源3D分割方法上的效果提出质疑，且忽视了我们在数据开源上的贡献。为此，我们整理了在更大版本的ctooth数据集上，开源3D分割方法的实验结果，用于辅助论证，查看链接：[google链接0521]。如果对项目有任何疑问或需要进一步讨论，欢迎通过GitHub与我们联系，期待与大家共同推动牙齿开源数据和算法社区的良性发展。

## 目录
- [数据准备](#数据准备)
- [环境搭建](#环境搭建)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [实验分析](#实验分析)
- [代码结构说明](#代码结构说明)
- [贡献指南](#贡献指南)
- [联系我们](#联系我们)

## 数据准备
1. **下载数据集**：从STS-3D 挑战赛下载ctooth数据集，包含原始CBCT图像及对应标注。
2. **数据预处理**：
    - 使用`data_preprocess.py`进行数据预处理，包括图像归一化、裁剪、重采样等操作，将数据处理为模型可接受的格式。
    - 执行命令：`python data_preprocess.py --input_dir [原始数据目录] --output_dir [处理后数据目录]`

## 环境搭建
1. **安装依赖**：
    - 确保已安装Python 3.7及以上版本。
    - 使用`pip install -r requirements.txt`安装项目所需的Python依赖库，主要包括PyTorch、NumPy、OpenCV等。
2. **配置GPU环境**（可选）：如果使用GPU进行训练和测试，请确保已正确安装CUDA和cuDNN，并在代码中正确配置相关参数。

## 模型训练
1. **训练参数配置**：在`train_config.py`文件中配置训练参数，包括训练批次大小、学习率、训练轮数、使用的注意力模块等。
2. **开始训练**：
    - 执行命令：`python train.py --config train_config.py`
    - 训练过程中，模型训练日志将保存在`logs`目录下，训练过程中的模型权重会定期保存到`checkpoints`目录。

## 模型测试
1. **测试参数配置**：在`test_config.py`文件中配置测试参数，指定测试数据路径、加载的模型权重路径等。
2. **开始测试**：
    - 执行命令：`python test.py --config test_config.py`
    - 测试完成后，分割结果将保存在`results`目录下，同时会生成相关的性能评估指标报告。

## 实验分析
1. **性能指标计算**：安装并调用surfdist package。

```
 def asd(mask_pred, mask_gt):
    import surface_distance as surfdist
    mask_gt = mask_gt.astype(np.bool)
    mask_pred = mask_pred.astype(np.bool)
    surface_distances = surfdist.compute_surface_distances(mask_gt, mask_pred, spacing_mm=(0.25, 0.25, 1))
    avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances)

    return (avg_surf_dist[0] + avg_surf_dist[1]) / 2


def hd(mask_pred, mask_gt):
    import surface_distance as surfdist
    mask_gt = mask_gt.astype(np.bool)
    mask_pred = mask_pred.astype(np.bool)
    surface_distances = surfdist.compute_surface_distances(mask_gt, mask_pred, spacing_mm=(0.25, 0.25, 1))
    hd_dist_95 = surfdist.compute_robust_hausdorff(surface_distances, 95)
    return hd_dist_95


def so(mask_pred, mask_gt):
    import surface_distance as surfdist
    mask_gt = mask_gt.astype(np.bool)
    mask_pred = mask_pred.astype(np.bool)
    surface_distances = surfdist.compute_surface_distances(mask_gt, mask_pred, spacing_mm=(0.25, 0.25, 1))
    surface_overlap = surfdist.compute_surface_overlap_at_tolerance(surface_distances, 1)
    return (surface_overlap[0] + surface_overlap[1]) / 2


def sd(mask_pred, mask_gt):
    import surface_distance as surfdist
    mask_gt = mask_gt.astype(np.bool)
    mask_pred = mask_pred.astype(np.bool)
    surface_distances = surfdist.compute_surface_distances(mask_gt, mask_pred, spacing_mm=(0.25, 0.25, 1))
    surface_dice = surfdist.compute_surface_dice_at_tolerance(surface_distances, 1)
    return surface_dice
```



3. **可视化分析**：利用`visualize.py`脚本对分割结果进行可视化，直观展示模型的分割效果。
    - 执行命令：`python visualize.py --result_dir [结果目录] --image_dir [原始图像目录]`
4. **对比实验**：在`comparison_experiments.py`中复现了当时业界几个开源的3D分割方法，并进行对比实验，可通过修改配置文件运行不同的对比实验。

## 代码结构说明(本项目代码在开源框架medzoo基础上进行开发)
ctooth_segmentation/
├── datasets/ # 存放数据
├── lib/ # 存放代码
│ ├── medloaders/ # 存放输出预处理类
│ │ ├── cbct.py
│ │ ├── cbct_utils.py
│ ├── medzoo/ # 存放模型文件
│ │ ├──UNet3D.py
│ │ ├── ...
│ ├── losses3D/ # 存放各类损失
│ │ ├── dice.py
│ │ ├── ...
│ ├── train/ # 存放训练和推理所需的类和函数
│ │ ├── trainer.py
│ │ ├── ...
│ ├── visual3D_temp #存放log及可视化函数
│ │ ├── BaseWriter.py
└── train_cbct.py # 主函数

## 贡献指南
欢迎大家对本项目提出宝贵意见或贡献代码。如果发现问题或有新的想法，可通过提交GitHub Issue进行反馈。如果想要贡献代码，请先fork本仓库，在本地修改后提交Pull Request，并详细说明修改内容和原因。

## 联系我们
如果你对项目有任何疑问、建议或合作意向，欢迎通过GitHub的Issue或私信与我们联系。



模型名称及链接列表
VNet：https://github.com/Dawn90/V-Net.pytorch
Unet3D：https://arxiv.org/abs/1606.06650（论文链接，代码复现基于 GitHub 开源实现）
HighResNet：https://arxiv.org/pdf/1707.01992.pdf（论文链接，代码复现基于 GitHub 开源实现）
DenseVoxelNet：https://arxiv.org/abs/1708.00573（论文链接，代码复现基于 GitHub 开源实现）

以上模型的复现均基于 GitHub 公开开源代码，其中部分模型（如 VNet）直接引用 GitHub 仓库中的实现，其余模型（如 Unet3D、HighResNet、DenseVoxelNet）参考论文理论并复用 GitHub 上的开源算法框架完成复现。
