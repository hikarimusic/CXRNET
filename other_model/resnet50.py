import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torchvision.models import resnet50, ResNet50_Weights
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.getcwd()), "data")
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
LOSS_FN = nn.MSELoss()
OPTIMIZER = torch.optim.SGD
LEARN_RATE = 0.1
LOAD_WEIGHT = None
SAVE_WEIGHT = "weight.pth"
CLASS = ["Atelectasis","Cardiomegaly","Consolidation","Edema","Effusion","Emphysema","Fibrosis","Hernia",
         "Infiltration","Mass","Nodule","Pleural_Thickening","Pneumonia","Pneumothorax"]
CLASS_NUM = 14
THRESHOLD = 0.99

class CustomDataset(Dataset):
    def __init__(self):
        self.table = pd.read_csv(os.path.join(DATA_DIR, "table.csv"))
        with open(os.path.join(DATA_DIR, "train.txt"), 'r') as f:
            self.train = f.read().split('\n')
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        img_name = self.train[idx]
        img_path = os.path.join(self.imagedir(img_name), img_name)
        image = read_image(img_path, mode=ImageReadMode.RGB)
        image = self.transform(image)
        label = self.table.loc[self.table["Image Index"]==img_name,"Finding Labels"].iloc[0]
        label = self.target_transform(label)
        return image, label 
    
    def imagedir(self, img):
        if img <= "00001335_006.png":
            return os.path.join(DATA_DIR, "image", "images_001", "images")
        elif img <= "00003923_013.png":
            return os.path.join(DATA_DIR, "image", "images_002", "images")
        elif img <= "00006585_006.png":
            return os.path.join(DATA_DIR, "image", "images_003", "images") 
        elif img <= "00009232_003.png":
            return os.path.join(DATA_DIR, "image", "images_004", "images")
        elif img <= "00011558_007.png":
            return os.path.join(DATA_DIR, "image", "images_005", "images") 
        elif img <= "00013774_025.png":
            return os.path.join(DATA_DIR, "image", "images_006", "images")
        elif img <= "00016051_009.png":
            return os.path.join(DATA_DIR, "image", "images_007", "images") 
        elif img <= "00018387_034.png":
            return os.path.join(DATA_DIR, "image", "images_008", "images")
        elif img <= "00020945_049.png":
            return os.path.join(DATA_DIR, "image", "images_009", "images") 
        elif img <= "00024717_000.png":
            return os.path.join(DATA_DIR, "image", "images_010", "images")
        elif img <= "00028173_002.png":
            return os.path.join(DATA_DIR, "image", "images_011", "images") 
        elif img <= "00030805_000.png":
            return os.path.join(DATA_DIR, "image", "images_012", "images")

    def transform(self, image):
        image = image.float()
        image /= 255
        return self.preprocess(image)   
    
    def target_transform(self, label):
        onehot = torch.zeros(14)
        label = label.split('|')
        for l in label:
            if l != "No Finding":
                onehot[CLASS.index(l)] = 1
        return onehot
        
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x

def Score(predict, answer):
    torch.set_printoptions(threshold=10_000)
    print("raw\n", predict)
    predict = torch.where(predict>THRESHOLD, 1., 0.)
    wrong = torch.where(predict!=answer, 1., 0.)

    print("pred\n", predict)
    print("answ\n", answer)

    predict = predict.sum(axis=1)
    answer = answer.sum(axis=1)
    wrong =  wrong.sum(axis=1)

    point = torch.zeros_like(answer)
    point = torch.where((answer<0.1) & (predict<0.1), 1., point)
    point = torch.where((answer>0.9) & (answer>wrong+0.1), (answer-wrong)/answer, point)
    return point

def train(data, model):
    size = len(data.dataset)
    model.train()
    for batch, (X, y) in enumerate(data):
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = model(X)[:,:14]
        loss = LOSS_FN(pred, y)
        optimizer = OPTIMIZER(model.parameters(), lr=LEARN_RATE)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        if SAVE_WEIGHT:
            torch.save(model.state_dict(), SAVE_WEIGHT)

        #if batch % 100 == 0:
        #    loss, current = loss.item(), batch * len(X)
        #    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def eval(data, model):
    size = len(data.dataset)
    num_batches = len(data)
    model.eval()
    loss, correct = 0, 0
    i = 0
    with torch.no_grad():
        for X, y in data:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)[:,:14]
            loss += LOSS_FN(pred, y).item()
            correct += Score(pred, y).sum().item()
            i += 1
            print(f"Error: \n Accuracy: {(100*correct/(len(X)*i)):>0.1f}%")
    loss /= num_batches
    correct /= size
    print(f"Eval Error: \n Accuracy: {(100*correct):>0.1f}%, Avg Loss: {loss:>8f} \n")

if __name__ == '__main__' :
    data = CustomDataset()
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    model = Model()
    if LOAD_WEIGHT:
        model.load_state_dict(torch.load(LOAD_WEIGHT))

    #for t in range(EPOCHS):
    #    print(f"\nEpoch {t+1}\n-------------------------------")
    #    train(dataloader, model)
    #    eval(dataloader, model)    

    eval(dataloader, model) 

