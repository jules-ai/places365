import torch
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import urllib.request
from PIL import Image

arch = 'resnet50'

model_file = f'{arch}_places365.pth.tar'
if not os.path.exists(model_file):
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
if not os.path.exists(file_name):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    print(f"Downloading {file_name}...")
    urllib.request.urlretrieve(synset_url, file_name)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

img_name = 'cam_example.jpg'
if not os.path.exists(img_name):
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
