# Places365 场景分类演示文档 (ResNet50)

本文档总结了使用 Places365 场景分类模型（特指 `resnet50_places365.pth.tar` 模型）时的前后处理步骤，并提供了一个最精简的、包含自动下载功能的 Python 演示脚本。

## 来源与资源链接

*   **原始 GitHub 仓库:** [https://github.com/csailvision/places365](https://github.com/csailvision/places365)
*   **模型下载路径 (ResNet50):** [http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar](http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar)

## 1. 预处理步骤 (Preprocessing)

**判断来源依据：** 预处理步骤的核心逻辑来源于本工程根目录下的 `run_placesCNN_basic.py` 文件（第31行至第36行）。

在将图像输入到 ResNet50 模型进行分类之前，必须对图像进行一系列转换，以匹配模型训练时的数据格式：

1.  **Resize (缩放)**: 首先将图像的长宽缩放到 256 像素 (`trn.Resize((256, 256))`)。
2.  **CenterCrop (中心裁剪)**: 从缩放后的图像中心裁剪出一个 224x224 的正方形区域 (`trn.CenterCrop(224)`)。这一步确保了输入尺寸符合 ResNet 模型的预期。
3.  **ToTensor (转为张量)**: 将裁剪后的图像（通常是 PIL Image 格式）转换为 PyTorch 张量，同时将像素值从 [0, 255] 归一化到 [0.0, 1.0]。
4.  **Normalize (标准化)**: 使用 ImageNet 数据集的均值和标准差对张量进行标准化处理。
    *   均值 (Mean): `[0.485, 0.456, 0.406]`
    *   标准差 (Std): `[0.229, 0.224, 0.225]`
5.  **增加 Batch 维度**: PyTorch 模型期望的输入形状为 `(batch_size, channels, height, width)`。因此，对于单张推理图像，需要使用 `.unsqueeze(0)` 增加一个大小为 1 的批次(batch)维度。

## 2. 后处理步骤 (Postprocessing)

模型完成前向传播后，会输出对应 365 个场景类别的原始 logits 值。为了解释这些预测结果，我们需要进行以下后处理：

1.  **Softmax**: 在维度 1 上对原始 logits 应用 Softmax 函数 (`F.softmax(logit, 1)`)。这将 logits 转换为 365 个类别的概率分布（所有概率之和为 1）。
2.  **Squeeze (降维)**: 移除大小为 1 的 Batch 维度 (`.squeeze()`)，使其变成一个一维的概率张量。
3.  **Sort (排序)**: 将概率值按降序排列 (`h_x.sort(0, True)`)。这会返回两个张量：
    *   `probs`: 排序后的概率值。
    *   `idx`: 这些概率值对应的原始索引（即类别索引）。
4.  **Label Mapping (标签映射)**: 取出概率最高的前几个 `idx`，并在 `categories_places365.txt` 文件中查找并映射出人类可读的场景类别名称。

## 3. 最精简 Demo 脚本 (`demo.py`)

以下是一个自包含的 Python 脚本。如果本地缺少模型权重、类别名称文件或测试图片，脚本会自动进行下载。

```python
import torch
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import urllib.request
from PIL import Image

# 使用的模型架构
arch = 'resnet50'

# 1. 自动下载并加载预训练模型权重
model_file = f'{arch}_places365.pth.tar'
if not os.path.exists(model_file):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    print(f"Downloading {model_file}...")
    urllib.request.urlretrieve(weight_url, model_file)

# 初始化模型
model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
# 移除 DataParallel 导致的 'module.' 前缀
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()

# 2. 定义预处理转换 (依据 run_placesCNN_basic.py)
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. 下载并加载类别标签映射文件
file_name = 'categories_places365.txt'
if not os.path.exists(file_name):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    print(f"Downloading {file_name}...")
    urllib.request.urlretrieve(synset_url, file_name)

classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

# 4. 下载测试图片并进行预处理
img_name = 'cam_example.jpg'
if not os.path.exists(img_name):
    img_url = 'http://places.csail.mit.edu/demo/' + img_name
    print(f"Downloading {img_name}...")
    urllib.request.urlretrieve(img_url, img_name)

img = Image.open(img_name)
# 增加 Batch 维度
input_img = centre_crop(img).unsqueeze(0)

# 5. 前向推理与后处理
with torch.no_grad():
    logit = model(input_img)

# 应用 Softmax 并移除 Batch 维度
h_x = F.softmax(logit, 1).squeeze()
# 按概率降序排序
probs, idx = h_x.sort(0, True)

# 6. 打印输出前五名预测结果
print(f'{arch} prediction on {img_name}')
for i in range(0, 5):
    print('{:.3f} -> {}'.format(probs[i].item(), classes[idx[i].item()]))
```
