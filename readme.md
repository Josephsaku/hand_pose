# 基于图文融合与交叉注意力的手部姿态估计

这是一个探索性项目，旨在通过结合**计算机视觉**与**自然语言处理**技术，实现对图像中手部姿态的3D坐标预测。项目的核心思想是利用**交叉注意力（Cross-Attention）机制**，将全局的文本指令与图像特征进行有效融合，从而引导模型更精确地理解和定位手部关节点。

本项目使用 **PyTorch** 框架进行搭建，并在经典的 **FreiHAND** 数据集上进行训练和验证。

## 项目成果

经过初步训练后，模型已能够从一张二维的手部图片中，预测出21个关键点的三维空间坐标，并成功将预测结果可视化为3D手部骨架。

## 当前状态

目前，项目已经完成了从数据处理、模型搭建、训练到预测可视化的完整流程。

<img src="plot_2025-09-25 23-40-05_0.png" alt="预测结果示例" width="600"/>

然而，正如上图所示，当前模型的预测精度还有很大的提升空间，主要不足体现在两个方面：

1.  **视觉特征提取能力有限**：当前模型使用了非常简单的自定义CNN作为图像编码器，其特征提取能力较弱，导致模型难以捕捉精细的手指细节和复杂的空间关系。这是导致预测结果不够自然的核心原因。
2.  **训练尚不充分**：为了快速验证流程，模型仅进行了少量的迭代训练（10个Epochs）。对于FreiHAND这样复杂的数据集，需要更长时间的训练才能让模型充分学习和收敛。

## 如何运行

1.  **环境配置**:
    ```bash
    # 建议使用Conda创建环境
    conda create -n hand_pose python=3.10
    conda activate hand_pose
    # 安装核心依赖
    conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
    conda install matplotlib
    ```

2.  **数据准备**:
    *   下载FreiHAND数据集 (`training` 和 `evaluation` 部分)。
    *   按照 `dataset.py` 文件中的预期，将数据整理到 `data/FreiHAND/` 目录下，并划分为 `training`, `validation`, `testing` 三个子文件夹。

3.  **训练模型**:
    ```bash
    python main.py
    ```

4.  **测试与可视化**:
    ```bash
    python predict.py
    ```
