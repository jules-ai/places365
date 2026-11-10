# Places365 Demo (ResNet50)

This document provides an overview of the preprocessing and postprocessing steps used in the Places365 scene classification model (specifically, the `resnet50_places365.pth.tar` model). It also provides a minimal, self-contained Python script to demonstrate inference.

## 1. Preprocessing Steps

Before feeding an image to the ResNet50 model for scene classification, the image must undergo several transformations to match the format the model was trained on:

1. **Resize**: The image is first resized such that both its width and height are 256 pixels (`trn.Resize((256, 256))`).
2. **CenterCrop**: A 224x224 square patch is cropped from the center of the resized image (`trn.CenterCrop(224)`). This ensures the input size matches the expected input size of ResNet models.
3. **ToTensor**: The cropped image (typically a PIL Image) is converted to a PyTorch tensor. The pixel values, originally in the range [0, 255], are scaled to [0.0, 1.0].
4. **Normalize**: The tensor is normalized using the mean and standard deviation from the ImageNet dataset.
   - Mean: `[0.485, 0.456, 0.406]`
   - Std: `[0.229, 0.224, 0.225]`
5. **Batch Dimension**: PyTorch models expect input in the shape `(batch_size, channels, height, width)`. Therefore, a batch dimension of size 1 is added to the single image tensor using `.unsqueeze(0)`.

These steps can be verified in `run_placesCNN_basic.py` (lines 31-36).

## 2. Postprocessing Steps

After the model performs a forward pass, it outputs raw logits for all 365 scene categories. To interpret these predictions, we perform the following postprocessing steps:

1. **Softmax**: A Softmax function is applied to the raw logits across dimension 1 (`F.softmax(logit, 1)`). This converts the logits into a probability distribution over the 365 classes, where all probabilities sum to 1.
2. **Squeeze**: The batch dimension (which was 1) is removed (`.squeeze()`), leaving a 1D tensor of probabilities.
3. **Sort**: The probabilities are sorted in descending order (`h_x.sort(0, True)`). This returns two tensors:
   - `probs`: The sorted probability values.
   - `idx`: The original indices of those probabilities (which correspond to class indices).
4. **Label Mapping**: The top indices from `idx` are used to look up the human-readable scene category names from the `categories_places365.txt` file.

## 3. Minimal Demo Script (`demo.py`)

The following is a self-contained Python script that implements the steps described above. It will automatically download the `resnet50_places365.pth.tar` model weights, the class categories file, and a sample image if they are not present.

```python
import torch
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import urllib.request
from PIL import Image

arch = 'resnet50'

model_file = f'{arch}_places365.pth.tar'
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    print(f"Downloading {model_file}...")
    urllib.request.urlretrieve(weight_url, model_file)

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()

centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

file_name = 'categories_places365.txt'
if not os.access(file_name, os.W_OK):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    print(f"Downloading {file_name}...")
    urllib.request.urlretrieve(synset_url, file_name)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

img_name = 'cam_example.jpg'
if not os.access(img_name, os.W_OK):
    img_url = 'http://places.csail.mit.edu/demo/' + img_name
    print(f"Downloading {img_name}...")
    urllib.request.urlretrieve(img_url, img_name)

img = Image.open(img_name)
input_img = centre_crop(img).unsqueeze(0)

with torch.no_grad():
    logit = model(input_img)
h_x = F.softmax(logit, 1).squeeze()
probs, idx = h_x.sort(0, True)

print(f'{arch} prediction on {img_name}')
for i in range(0, 5):
    print('{:.3f} -> {}'.format(probs[i].item(), classes[idx[i].item()]))
```
