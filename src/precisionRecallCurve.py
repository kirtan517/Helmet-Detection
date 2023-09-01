import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from torch.utils.data import DataLoader
from loader import AICITY2023TRACK5
import os
import tqdm
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIRECTORY = os.path.join(os.getcwd(),"data")
TRAIN_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY,"aicity2023_track5",)
IMAGE_VAL_DIRECTORY = os.path.join(TRAIN_DATA_DIRECTORY,"validationImages")

from model import initialize_model
model = initialize_model("efficientnet",8,False,False)
checkpoints = torch.load(os.path.join(os.getcwd(),"src","best_model.pth"))
model.load_state_dict(checkpoints)

validationDataset = AICITY2023TRACK5(TRAIN_DATA_DIRECTORY,IMAGE_VAL_DIRECTORY,validation=True)
valLoader = DataLoader(validationDataset,batch_size = 64,shuffle = True,collate_fn=validationDataset.collate_fn)

softmax = torch.nn.Softmax()

def decode(prediction):
    return torch.max(prediction,dim = 1)

def computePredictions(model,validationLoader):
    model.eval()
    model.to(DEVICE)
    ground_truth,predictions = [],[]
    for batch in tqdm.tqdm(validationLoader):
        x,y = batch
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        prediction = model(x)
        # prediction,_ = decode(prediction)
        prediction = softmax(prediction)
        ground_truth.extend(y.detach().cpu().numpy().tolist())
        predictions.extend(prediction.detach().cpu().numpy().tolist())

    return np.array(ground_truth),np.array(predictions)

ground_truth, predictions = computePredictions(model,valLoader)
print(predictions.shape)

precision = {}
recall = {}
categories = {0 : "motorbike",1 : "DHelmet",2 : "DNoHelmet",3 : "P1Helmet",4 : "P1NoHelmet",5 :  "P2Helmet",6: "P2NoHelmet", 7 : "Other"}
n_classes = 8  # number of classes in your multi-class problem
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(ground_truth == i, predictions[:,i])

plt.figure(figsize=(10, 10))
for i in range(n_classes):
    plt.plot(recall[i], precision[i], lw=2, label=f'Class {categories[i]}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.title('Precision-Recall Curve for Multi-Class Classification')
plt.show()