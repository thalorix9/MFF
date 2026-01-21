# Medical Multi-Focus Image Fusion (MFF)

本本科毕业设计项目旨在处理医学影像的多焦图像融合。项目包含完整的图像处理管线：自动图像配准（修正微抖动）与基于清晰度评价的图像融合。

## ✨ 功能特性

*   **自动配准 (Auto Alignment)**: 使用 ECC (Enhanced Correlation Coefficient) 算法，自动修正多张拍摄图像之间的微小位移（平移、旋转、仿射变换）。
*   **智能融合 (Smart Fusion)**: 基于拉普拉斯能量 (Laplacian Energy) 的清晰度评价，自动选取每张图中最清晰的像素区域。
*   **无损视野 (Lossless FOV)**: 引入 Alpha 透明通道处理配准边界，避免传统裁剪方法造成的视野损失，同时消除边缘黑边伪影。
*   **批量处理**: 支持一键处理多个分组的实验数据。

## 📂 项目结构

```text
MFF/
├── main.py                 # 主程序入口，负责数据加载与流程控制
├── setup_env.bat           # 环境一键配置脚本 (Windows)
├── run.bat                 # 一键运行脚本 (Windows)
├── environment.yml         # Conda 环境配置文件
├── .gitignore              # Git 忽略配置
├── README.md               # 项目说明文档
├── src/                    # 核心算法源码
│   ├── align.py            # 图像配准模块 (ECC + BGRA处理)
│   └── fuse.py             # 图像融合模块 (Laplacian + Mask融合)
├── UnregisteredImages/     # [数据目录] 原始输入图片 (按组分类)
│   ├── Group1/             #   示例：第一组多焦图片
│   └── Group2/             #   示例：第二组多焦图片
└── Results/                # [输出目录] 存放融合结果
    ├── Group1/
    └── Group2/
```

## 🚀 快速开始

### 1. 环境准备
本项目使用 Miniconda 进行环境管理。

**方式 A (推荐)**:
双击运行 `setup_env.bat`，脚本将自动创建名为 `mff_env` 的虚拟环境并安装依赖。

**方式 B (手动)**:
```bash
conda env create -f environment.yml
conda activate mff_env
```

### 2. 准备数据
将您的多焦图像放入 `UnregisteredImages` 目录下。请为每一组实验创建一个子文件夹。
例如：
```text
UnregisteredImages/
  ├── Patient_001/
  │   ├── focus_1.jpg
  │   └── focus_2.jpg
  └── Patient_002/
      ├── 1.png
      └── 2.png
```

### 3. 运行代码
双击 `run.bat` 或在终端运行：
```bash
python main.py
```

程序将自动扫描所有子文件夹，进行配准和融合，并将结果保存到 `Results/` 对应目录中。

## 🛠️ 核心算法说明

1.  **配准阶段 (`src/align.py`)**:
    *   将图像转换为 4 通道 BGRA 格式。
    *   使用 `cv2.findTransformECC` 计算亚像素级变换矩阵。
    *   使用 `cv2.warpAffine` 对齐图像，边界填充为透明 (0,0,0,0)。

2.  **融合阶段 (`src/fuse.py`)**:
    *   计算每张图的拉普拉斯梯度图作为清晰度指标。
    *   使用高斯模糊平滑决策边界。
    *   通过 Alpha 通道掩码过滤无效区域（边缘伪影）。
    *   逐像素选取最大清晰度值的来源，合成最终图像。