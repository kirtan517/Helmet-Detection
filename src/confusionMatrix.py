import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from torch.utils.data import DataLoader
from loader import AICITY2023TRACK5
import os
import tqdm

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

def decode(prediction):
    return torch.argmax(prediction,dim = 1)

def computeConfusionMatrix(model,validationLoader):
    model.eval()
    model.to(DEVICE)
    ground_truth,predictions = [],[]
    for batch in tqdm.tqdm(validationLoader):
        x,y = batch
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        prediction = model(x)
        prediction = decode(prediction)
        ground_truth.extend(y.detach().cpu().numpy().tolist())
        predictions.extend(prediction.detach().cpu().numpy().tolist())

    return np.array(ground_truth),np.array(predictions)

ground_truth,predicitons = computeConfusionMatrix(model,valLoader)

cm = confusion_matrix(ground_truth, predicitons)
# sns.set (rc = {'figure.figsize':(18, 18)})
labels = ["motorbike","DHelmet","DNoHelmet","P1Helmet","P1NoHelmet","P2Helmet","P2NoHelmet","Other"]
fig,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(cm,annot = True, fmt="d", cmap="Blues", cbar=False,xticklabels=labels,yticklabels=labels)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix')
# ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels);
plt.savefig("ConfusionMatrix.png")
plt.show()
