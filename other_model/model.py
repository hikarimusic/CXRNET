import torch
from torch import nn
from torchvision import transforms, io

IMAGE_SIZE = 1024*1024
CLASS = ["Atelectasis","Cardiomegaly","Consolidation","Edema","Effusion","Emphysema","Fibrosis","Hernia",
         "Infiltration","Mass","Nodule","Pleural_thickening","Pneumonia","Pneumothorax","No Finding"]


class resnet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.preprocess = nn.Sequential(
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

    def forward(self, x):
        x = self.preprocess(x)
        x = self.model(x)
        print(x.shape)
        print(x)
        print(torch.sum(x))
        return x



'''
im = io.read_image("sample1.png", mode=io.ImageReadMode.RGB)
im = im.float()
im /= 255
im = im[None,:]

a = resnet50()
a.forward(im)
'''



'''
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
input_image = Image.open('dog.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)
a = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
output = a(input_batch)
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)

with open("classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
'''

