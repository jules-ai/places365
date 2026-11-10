import torch
from torchvision import models, transforms as trn
from torch.nn import functional as F
from PIL import Image

def demo_resnet50_places365(img_path, model_file='resnet50_places365.pth.tar', categories_file='categories_places365.txt'):
    # 1. 加载模型结构和预训练权重
    model = models.resnet50(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)

    # 兼容 DataParallel 保存的模型权重，去掉 'module.' 前缀
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    # 2. 前处理 (Preprocessing)
    # 标准的 ImageNet 前处理方式：缩放 -> 中心裁剪 -> 转 Tensor -> 归一化
    preprocess = trn.Compose([
        trn.Resize((256, 256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0) # 增加 batch 维度

    # 3. 推理 (Inference)
    with torch.no_grad():
        logit = model(input_tensor)

    # 4. 后处理 (Postprocessing)
    # 进行 softmax 获取分类概率
    probs = F.softmax(logit, dim=1).squeeze()

    # 对概率进行降序排序
    topk_probs, topk_idx = probs.sort(dim=0, descending=True)

    # 加载类别标签
    with open(categories_file, 'r') as f:
        # categories_places365.txt 格式: /a/airfield 0 -> 取 "/a/airfield" 并去掉前三位，结果为 "airfield"
        classes = [line.strip().split(' ')[0][3:] for line in f]

    # 输出 Top-5 预测结果
    print(f'Top-5 predictions for {img_path}:')
    for i in range(5):
        print(f'{topk_probs[i]:.3f} -> {classes[topk_idx[i]]}')

if __name__ == '__main__':
    # 示例运行 (需确保相关文件存在，本代码仅作静态展示)
    demo_resnet50_places365('test.jpg')
