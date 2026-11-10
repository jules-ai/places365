# Places365 ResNet50 推理 Demo 说明

本说明文档总结了使用 `resnet50_places365.pth.tar` 模型进行场景分类时的前后处理步骤和方法，并配套提供了一个最精简的 Demo 脚本（`demo_resnet50.py`）。

## 前处理 (Preprocessing)

前处理的目的是将输入的原始图片转换为模型能够接受的标准 Tensor 格式，步骤遵循标准的 ImageNet 预处理流程：

1. **Resize (缩放)**: 将图片缩放，使短边等于 256 像素 (`Resize((256, 256))`)。
2. **CenterCrop (中心裁剪)**: 从图片中心裁剪出 224x224 像素的区域 (`CenterCrop(224)`)。
3. **ToTensor (转张量)**: 将 PIL 图片或 NumPy 数组转换为 PyTorch Tensor，并将像素值从 `[0, 255]` 归一化到 `[0.0, 1.0]` 区间。
4. **Normalize (标准化)**: 使用固定的均值和标准差对每个通道（RGB）进行标准化。
    - **Mean (均值)**: `[0.485, 0.456, 0.406]`
    - **Std (标准差)**: `[0.229, 0.224, 0.225]`
5. **Unsqueeze (增加维度)**: 模型输入需要 Batch 维度，因此需要使用 `unsqueeze(0)` 将 shape 从 `[C, H, W]` 扩展为 `[1, C, H, W]`。

## 后处理 (Postprocessing)

后处理的目的是将模型输出的 Logits（未归一化的原始预测值）转换为直观的概率和人类可读的类别标签：

1. **Softmax**: 对模型输出的 logits 在类别维度（`dim=1`）上应用 Softmax 函数，将输出映射为 `[0, 1]` 区间内的概率值，且概率总和为 1。
2. **Squeeze**: 移除批次维度，使结果变成一维向量，方便后续处理。
3. **Sort (排序)**: 将概率值按降序排列，获取概率最高的前 K 个（Top-K）概率值及其对应的索引。
4. **类别映射**: 解析 `categories_places365.txt` 文件（格式如：`/a/airfield 0`），根据上一步提取的索引找到对应的场景类别名称。
5. **输出结果**: 打印出 Top-K（例如 Top-5）的预测概率和对应的类别名称。

## 如何使用 Demo 脚本

`demo_resnet50.py` 是一个不需要额外下载文件的精简脚本，用于静态展示如何加载模型并进行前后处理。

**前置条件**:
- 安装 `torch` 和 `torchvision`。
- 确保当前目录下存在以下文件：
    - `resnet50_places365.pth.tar`: 模型权重文件。
    - `categories_places365.txt`: 类别标签文件。
    - `test.jpg`: 用于测试的输入图片。

**运行方式**:
（此脚本仅供静态审查参考，如果上述文件已备齐可直接运行）
```bash
python demo_resnet50.py
```
