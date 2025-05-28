
## 项目简介
在我们的工作之前，用作研究用途的高质量3D牙齿CBCT数据几乎没有开源，这一情况极大限制了当时的牙齿3D分割算法的研究。为解决这一痛点，我们花费大量精力收集和标注牙齿CBCT数据，并随着2023年的MICCAI挑战赛进行了开源，以证明我们数据开源工作的真实性和贡献。

同时，为了方便后续研究者将ctooth数据用于自动化牙齿分割任务以辅助牙医诊疗场景，我们不得不复现在其他领域相对成熟的开源3D分割方法作为对照方法，并统计了它们在ctooth上的分割效果。实验结果表明，这些对照方法在ctooth上表现不足，主要原因如下：
- **缺乏针对性设计**：部分方法为其他3D分割任务或通用场景设计，缺乏针对牙齿3D小目标的架构优化（如DenseVoxelNet和3D HighResNet原用于MR心脏分割）
- **特征损失与过拟合**：参数量较大的模型（如3D U-Net、V-Net）深度通常为5层左右，且层间的3D Maxpooling操作易造成小牙齿局部特征损失

针对上述问题，我们提出了一个牙齿分割基准框架，提供以下功能：
- 丰富的3D CBCT预处理方式
- 多种损失函数计算实现
- 可扩展的3D分割模型库
- 批量超参数设置与实验管理
- 自动化3D分割性能评估

目前，我们已整理相关工作并开源供算法研究者参考。近期，有人在未与我们事先沟通的情况下，就对照实验效果提出质疑，忽视了我们在数据开源上的贡献。为此我们补充了更大版本ctooth数据集上的实验结果，可通过[google链接0521](google链接0521)查看。欢迎通过GitHub与我们联系，共同推动牙齿开源数据和算法社区的发展。我们深知当前工作仍有改进空间，若有考虑不周之处，也请各位前辈与同行不吝赐教（能力有限，轻喷哦～）。


## 目录
- [数据准备](#数据准备)
- [环境搭建](#环境搭建)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [实验分析](#实验分析)
- [代码结构说明](#代码结构说明)
- [贡献指南](#贡献指南)


## 数据准备

### 1. 下载数据集
从STS-3D挑战赛下载ctooth数据集，包含原始CBCT图像及对应标注。

### 2. 数据预处理
在`train_cbct.py`中使用`medical_loaders.generate_datasets()`进行预处理，核心功能包括：

```python
# 主要预处理流程
# 1. DICOM图像预处理
- 将DICOM格式图像转换为JPG/PNG格式
- 调整图像窗口（Windowing）增强对比度
- 支持多种归一化方法（标准化、最大值归一化等）

# 2. 掩码处理
- 填充掩码中的空洞（FillHole函数）
- 二值化处理和自适应阈值操作

# 3. 数据增强
- 添加高斯噪声（noise函数）
- 可配置噪声均值和标准差

# 4. 3D体数据生成
- 将2D切片组合为3D体积数据
- 支持自定义体积大小和重叠率

# 5. 数据分块
- 将大体积数据分割为小训练样本
- 支持训练集和验证集不同重叠策略

# 6. 数据保存
- 将处理后的数据保存为NPY格式
- 生成训练样本列表
```


## 环境搭建

### 1. 安装依赖
- 确保已安装Python 3.8及以上版本
- 安装必要的Python库：
  ```bash
  pip install torch numpy opencv-python pydicom surface-distance
  ```

### 2. 配置GPU环境（可选）
如需使用GPU训练，请确保：
1. 正确安装CUDA和cuDNN
2. 在代码中配置可用GPU：
   ```python
   # train_cbct.py
   os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"  # 指定使用第0和第1块GPU
   ```


## 模型训练

### 1. 训练参数配置
在`train_cbct.py`的`get_arguments()`函数中配置以下参数：
- `batch_size`: 训练批次大小
- `learning_rate`: 学习率
- `epochs`: 训练轮数
- `patch_size`: 3D Patch大小（深度×高度×宽度）

### 2. 开始训练
执行命令：
```bash
python train_cbct.py
```


## 模型测试

### 1. 测试参数配置
确保`train_cbct.py`中以下代码未被注释：
```python
# train_cbct.py
print("START ANALYSE...")
trainer.testing()
```

### 2. 执行测试
```bash
python train_cbct.py
```

### 3. 结果保存
- 分割结果将保存在模型同名文件夹下
- 性能评估指标将导出为Excel文件


## 实验分析

### 1. 性能指标计算
支持多种分割评估指标，包括：

```python
def iou_score(output, target):
# 计算交并比(IOU)
    smooth = 1e-5
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output &gt; 0.5
    target_ = target &gt; 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    return (intersection + smooth) / (union + smooth)


def sensitivity(output, target):
# 计算敏感度(Sensitivity)
    smooth = 1e-5
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    intersection = (output * target).sum()
    return (intersection + smooth) / \
           (target.sum() + smooth)


def ppv(output, target):
# 计算阳性预测值(PPV)
    smooth = 1e-5
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    intersection = (output * target).sum()
    return (intersection + smooth) / \
           (output.sum() + smooth)
```

表面距离相关指标（需安装`surface-distance`包，下方的函数定义在`trainer`中）：

```python

 def asd(mask_pred, mask_gt):
 # 计算平均表面距离(ASD)
    import surface_distance as surfdist
    mask_gt = mask_gt.astype(np.bool)
    mask_pred = mask_pred.astype(np.bool)
    surface_distances = surfdist.compute_surface_distances(mask_gt, mask_pred, spacing_mm=(0.25, 0.25, 1))
    avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances)

    return (avg_surf_dist[0] + avg_surf_dist[1]) / 2


def hd(mask_pred, mask_gt):
 # 计算Hausdorff距离(HD)

    import surface_distance as surfdist
    mask_gt = mask_gt.astype(np.bool)
    mask_pred = mask_pred.astype(np.bool)
    surface_distances = surfdist.compute_surface_distances(mask_gt, mask_pred, spacing_mm=(0.25, 0.25, 1))
    hd_dist_95 = surfdist.compute_robust_hausdorff(surface_distances, 95)
    return hd_dist_95


def so(mask_pred, mask_gt):
# 计算表面重叠率(SO)
    import surface_distance as surfdist
    mask_gt = mask_gt.astype(np.bool)
    mask_pred = mask_pred.astype(np.bool)
    surface_distances = surfdist.compute_surface_distances(mask_gt, mask_pred, spacing_mm=(0.25, 0.25, 1))
    surface_overlap = surfdist.compute_surface_overlap_at_tolerance(surface_distances, 1)
    return (surface_overlap[0] + surface_overlap[1]) / 2


def sd(mask_pred, mask_gt):
# 计算表面Dice系数(SD)
    import surface_distance as surfdist
    mask_gt = mask_gt.astype(np.bool)
    mask_pred = mask_pred.astype(np.bool)
    surface_distances = surfdist.compute_surface_distances(mask_gt, mask_pred, spacing_mm=(0.25, 0.25, 1))
    surface_dice = surfdist.compute_surface_dice_at_tolerance(surface_distances, 1)
    return surface_dice
    
```

### 2. 对比实验设置
支持扩展新模型，配置步骤：

1. 在`train_cbct.py`的参数列表中添加模型名称：
   ```python
   parser.add_argument('--model', type=str, default='UNet3D', choices=['UNet3D', 'TransBTS', ...])
   ```

2. 在`medzoo/__init__.py`中注册新模型：
   ```python
   model_list = ['UNet3D', 'TransBTS', ...]

   def create_model(args):
       # ... 其他模型定义
       elif model_name == 'TransBTS':
           _, model = TransBTS(args, dataset='brats', _conv_repr=True, _pe_type="learned")
   ```

3. 将模型实现文件（如`TransBTS.py`）放入`medzoo/`目录


支持在U型结构中替换不同Attention模块，配置步骤：
 
1. 选择基础U型架构  
   - 从现有模型中选取轻量级U型网络（如`Unet3DMINI`），或通过修改`3DUNet.py`减少网络深度（例如将编码层数从5层简化为3层），构建基础骨架。  

2. 优化编码器结构（可选）  
   - 在编码器（Encoder）的卷积块中引入残差连接（Residual Structure），增强特征传递能力。例如，将传统的`Conv3D+BN+ReLU`模块替换为残差模块：  

3. 在瓶颈层（Bottleneck）接入Attention模块 
   - 核心原则：多数论文中的Attention模块（如SE、CBAM、Non-Local等）具有即插即用特性，可直接迁移至3D网络。以`External Attention`为例（参考仓库：[xmu-xiaoma666/External-Attention-pytorch](https://github.com/xmu-xiaoma666/External-Attention-pytorch)）：  
     1. 在`medzoo`目录下添加Attention模块实现文件（如`attention3d.py`）；  
     2. 在Bottleneck层中导入并调用Attention模块
 
4. 新手参考路径
   - 初次尝试时，可先研读`medzoo`中已实现的Attention接入方式，理解3D特征图的维度适配（如`(B, C, D, H, W)`）后，再替换为目标模块。  


推荐资源与注意事项：  
- **参考仓库**：[External-Attention-pytorch](https://github.com/xmu-xiaoma666/External-Attention-pytorch)。  
- **维度适配**：3D Attention需注意卷积核维度（如`3×3×3`）和特征图维度（添加`dim=3`参数）。  
- **轻量化设计**：牙齿分割场景中，建议优先使用轻量级Attention，避免因参数量过大导致过拟合。

### 3. 前后处理注意事项
分块大小、重叠率以及重叠部分的协议机制对3D结构重建和后续性能评估有显著影响，在训练前请参考如下tips：
- 尽可能用更多更大memory的GPU训练模型，减少分块带来的边界效应，这种噪声影响在CBCT数据上尤为敏感，外部易感知到性能不稳定，或者实验结果的不易复制。
- 水平/垂直/深度方向重叠率：这个超参数在二维图像上易缓解边界效应，尤其是在重叠率增大的情况下，然而，对于3D CBCT图像中的牙齿目标，特别是牙根小目标区域，这种重叠可能在一些样本中会加重边界效应，因此针对不同实验条件，需要谨慎调参。
- 分块volume水平/垂直/深度分辨率：这个超参数会一定程度上影响分割性能，根据经验值，在硬件设备支持的情况下，分辨率设置的较高，生成的分割结果会越连续。
- 通过patch重建3D结构代码如下，默认使用二值化，可优化为更加柔和的手段，例如“少数服从多数”投票的方式，减少孤立噪声点和牙齿预测结果的不连续性。
```python
- for i in range(x.shape[0]):

                img = x[i]
                img[img != 0] = 255
                img[img!=255]=0
                img_rgb= cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)



                # print(gt_idx,j,i)
                ground_truth=gt[i]
                ground_truth[ground_truth != 0] = 255
                ground_truth[ground_truth!=255]=0
```

## 代码结构说明
本项目基于开源框架medzoo开发，代码结构如下：

```
ctooth_segmentation/
├── datasets/              # 数据集存放目录
├── lib/                   # 核心代码库
│   ├── medloaders/        # 数据加载与预处理
│   │   ├── cbct.py        # CBCT数据加载器
│   │   └── cbct_utils.py  # 辅助工具函数
│   ├── medzoo/            # 模型定义
│   │   ├── __init__.py    # 模型注册表
│   │   ├── UNet3D.py      # 3D U-Net实现
│   │   └── ...
│   ├── losses3D/          # 损失函数
│   │   ├── dice.py        # Dice损失
│   │   └── ...
│   ├── train/             # 训练与推理
│   │   ├── trainer.py     # 训练器
│   │   └── ...
│   └── visual3D_temp/     # 日志与可视化
│       └── BaseWriter.py  # 基础记录器
└── train_cbct.py          # 主训练脚本
```


## 模型实现说明
本项目复现并集成了多种3D分割模型，所有实现均基于公开开源代码：

- **VNet**：https://github.com/Dawn90/V-Net.pytorch  
- **Unet3D**：https://arxiv.org/abs/1606.06650  
- **HighResNet**：https://arxiv.org/pdf/1707.01992.pdf  
- **DenseVoxelNet**：https://arxiv.org/abs/1708.00573  

其中，VNet直接引用GitHub仓库实现，其余模型参考论文理论并复用开源框架完成复现。


## 贡献指南
欢迎参与项目贡献！请遵循以下流程：

1. 发现问题或有新想法时，提交GitHub Issue说明
2. 贡献代码时：
   - Fork本仓库到个人账号
   - 创建新分支进行开发
   - 提交Pull Request并详细说明修改内容

