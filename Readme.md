<a name="br1"></a>

YOLO v5 Model

In this approach all the images and labels were train on the yolov5m model.

As it can be seen from the correlaꢀon matrix that the model was unable to predict the

P1Helmet correctly at all of the predicꢀons for that class were wrong and same goes for

P2Helmet and P2NoHelmet as well.

<a name="br2"></a>

As it can be seen from the above ﬁgure the there is high class mismatch as there are small

number of bounding box which are being idenꢀﬁed as the P1Helmet , P2Helmet and

P2NoHelmet this might be causing the model to perform poorly on the P1Helmet classiﬁcaꢀon.

The major for such a low predicꢀon for P2 might be that having 3 people on the motorbike is a

rare event and further to classify that it’s the p2 model need to ﬁrst idenꢀfy that there is p1

present in the motorbike so such sort of sequenꢀal learning might be causing the model to

predict poorly on this parꢀcular class.

<a name="br3"></a>

As it can be seen from the above ﬁgure that mAP@05 is 0.8 mAP@0.5:0.95 is 0.6 and the above

model was trained for 15 epochs.

YOLOv5 CODE

<a name="br4"></a>

Result obtained on Test Dataset

<a name="br5"></a>

Exploratory Data Analysis And Preprocessing

As can be seen from the above curves the best height and width to be selected for resizing the

image would be around 100. And as can be seen on average most of the images have 2-4

objects.

EDA CODE

<a name="br6"></a>

<a name="br7"></a>

<a name="br8"></a>

<a name="br9"></a>

Using Neural Network as Classiﬁer

In this approach ﬁrstly a yolo network was used to idenꢀfy the diﬀerent object present in the

image which is motorbike and person. Then the results of the model which are the cropped

images of the bounding boxes detected for these object is passed to an eﬃcient net

architecture to classify if that cropped image is one of the seven class described in the

compeꢀꢀon. One extra class was added so as to make the model classify the cropped image as

other if none of those seven classes meet.

But during the training this eﬃcient net architecture was trained with the cropped images

obtained from the ground truth.

As the model uses pretrained weights but the parameters are not frozen during training thus it

requires few epochs to get used to the new dataset.

<a name="br10"></a>

As it can be seen from this confusion matrix that although the model is performing really well

sꢀll as there is not dataset for other category it will be diﬃcult for the model to classify the

cropped image into this one

<a name="br11"></a>

_Result obtained for test dataset_

<a name="br12"></a>

Note – in this approach yolo model was not trained at all and as yolo model was not trained at

all and the classiﬁer does not have any of the cropped images for the other category it

classifying each and every person bounding box predicted by the yolo model as either of those 7

classes

Now the above issue could have been solved by generaꢀng the custom dataset by passing it

thorough yolo and checking IoU with the ground truth values and if the IoU is equal to 0 then

that cropped image should have been marked as other and then the model should have been

trained on this dataset. But because of the ꢀme constrain I was not able to do that.

CODE

Loader.py

import pandas as pd

import os

import numpy as np

import matplotlib.pyplot as plt

import cv2

import IPython

from glob import glob

from torch.utils.data import Dataset,DataLoader

from sklearn.model_selection import train_test_split

import tqdm

import seaborn as sns

import albumentations as A

import torch

<a name="br13"></a>

def generateDataFrame(trainDataDirectory):

column_names = ["video_id", "frame", "track_id", "bb_left", "bb_top",

"bb_width","bb_height","category"]

df = pd.read_csv(os.path.join(trainDataDirectory,"gt.txt") , names =

column_names)

df["category"] = df["category"] - 1

df["image\_path"] = df["video\_id"].apply(lambda x : f"{x:03}")+ "\_"+

df["frame"].apply(lambda x : f"{x-1:03}") + ".jpeg"

df["normalized\_center\_x"] = (df["bb\_left"] + df["bb\_width"]//2) / 1920

df["normalized\_center\_y"] = (df["bb\_top"] + df["bb\_height"]//2) / 1080

df["normalized\_width"] = df["bb\_width"] / 1920

df["normalized\_height"] = df["bb\_height"] / 1080

df =

df[~df["image\_path"].isin(["034_200.jpeg","034_201.jpeg","034_202.jpeg","034\_

203\.jpeg","034_204.jpeg"])].reset_index()

split_on = df["image\_path"].unique()

train_split,val_split =

train_test_split(split_on,test_size=0.10,shuffle=True)

train_df = df[df["image\_path"].isin(train_split)]

val_df = df[df["image\_path"].isin(val_split)]

train_df = train_df.reset_index()

val_df = val_df.reset_index()

train_df.to_csv(os.path.join(trainDataDirectory,"train.csv"))

val_df.to_csv(os.path.join(trainDataDirectory,"val.csv"))

return train_df,val_df

def readDataFrame(trainDataDirectory):

if(os.path.isfile(os.path.join(trainDataDirectory,"train.csv")) and

os.path.isfile(os.path.join(trainDataDirectory,"val.csv"))):

train_df = pd.read_csv(os.path.join(trainDataDirectory,"train.csv"))

val_df = pd.read_csv(os.path.join(trainDataDirectory,"val.csv"))

else:

train_df,val_df = generateDataFrame(trainDataDirectory)

return train_df,val_df

class AICITY2023TRACK5(Dataset):

def

\_\_init\_\_(self,Training_directory,Training_image_directory,resized_width=100,r

esized_height=100,transform = None,validation = False):

_"""_

_path -> Training Directory_

_"""_

self.Training_directory = Training_directory

self.Training_image_directory = Training_image_directory

self.validation = validation

if validation:

\_ , self.df = readDataFrame(Training_directory)

else:

self.df, \_ = readDataFrame(Training_directory)

self.unique_images = self.df["image\_path"].unique()

self.labelsToClasses = {0 : "motorbike",1 : "DHelmet",2 :

"DNoHelmet",3 : "P1Helmet",4 : "P1NoHelmet",5 : "P2Helmet",6: "P2NoHelmet"}

self.resized_width = resized_width

self.resized_height = resized_height

self.transform = transform

<a name="br14"></a>

def \_\_len\_\_(self):

return len(self.unique_images)

def \_\_getitem\_\_(self,index):

image_name = self.unique_images[index]

image =

cv2.imread(os.path.join(self.Training_image_directory,image_name))

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

images,labels = self.processImage(image,self.df[self.df["image\_path"]

== image_name])

return {"images": images,"labels":labels,"image":image}

def processImage(self,image,bounding_boxs):

images = []

labels = []

\# print(image.shape) # 1080,1920,3

for index,bbounding_box in bounding_boxs.iterrows():

start_point = (bbounding_box["bb\_left"],bbounding_box["bb\_top"])

end_point =

(bbounding_box["bb\_width"]+start_point[0],bbounding_box["bb\_height"]+start_po

int[1])

finalImage =

image[start_point[1]:end_point[1],start_point[0]:end_point[0],:]

if self.transform:

finalImage = self.transform(image = finalImage)["image"]

images.append(finalImage)

labels.append(bbounding_box["category"])

return images,labels

def resize_image(self,image):

new_size = (self.resized_width, self.resized_height)

resized_img = cv2.resize(image, new_size)

return resized_img

def collate_fn(self,data):

labels = [j for i in data for j in i["labels"]]

images = [self.resize_image(j) for i in data for j in i["images"]]

images =

torch.permute(torch.tensor(images,dtype=torch.float),(0,3,1,2)) # Batch x

channels x height x width

labels = torch.tensor(labels,dtype=torch.long)

return images , labels

class AICITY2023TRACK5TEST(Dataset):

def

\_\_init\_\_(self,test_directory,test_image_directory,test_label_directory,testVa

lidationFlag = False,transform = None,resized_width =

<a name="br15"></a>

100,resized_height=100):

_"""_

_testValidationFlag -> true when the dataset is for test_

_"""_

self.directory = test_directory

self.test_image_directory = test_image_directory

self.testValidationFlag = testValidationFlag

self.test_label_directory = test_label_directory

self.resized_width = resized_width

self.resized_height = resized_height

if testValidationFlag:

images = os.listdir(test_image_directory)

images_dir = [os.path.join(test_image_directory,i) for i in

images]

images = [i.split(".")[0] for i in images]

labels = os.listdir(test_label_directory)

labelFinal = []

labelDirectory = []

for image in images:

label_name = image.split(".")[0] + ".csv"

if label_name in labels:

labelFinal.append(label_name)

labelDirectory.append(os.path.join(test_label_directory,label_name))

output = {"images" : images,"image_directory" : images_dir,

"labels" : labelFinal,"label_directory" :

labelDirectory,}

self.df = pd.DataFrame(output)

else:

file_path = os.path.join(test_directory,"val.csv")

if not os.path.isfile(file_path):

\_,self.df = readDataFrame(test_directory)

else:

self.df = pd.read_csv(file_path)

self.df["images"] = self.df["image\_path"].apply(lambda x :

x.split(".")[0])

self.df["image\_directory"] = self.df["image\_path"].apply(lambda x

: os.path.join(test_image_directory,x))

self.transform = transform

self.uniques = self.df["image\_directory"].unique()

def \_\_len\_\_(self):

return len(self.uniques)

def \_\_getitem\_\_(self,index):

self.image_path = self.uniques[index]

image = cv2.imread(self.image_path)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

if self.testValidationFlag:

\# This is the test thing

label = self.df[self.df["image\_directory"] == self.image_path]

labels = pd.read_csv(label["label\_directory"].item())

#labels should be in the list formate

<a name="br16"></a>

labels["label\_directory"] = label["label\_directory"].item()

images,labels = self.processImage(image,labels)

return {"images":images,"labels":labels}

else:

self.labels = self.df[self.df["image\_directory"] ==

self.image_path]

return {"image":image,"labels":self.labels}

def resize_image(self,image):

new_size = (self.resized_width, self.resized_height)

resized_img = cv2.resize(image, new_size)

return resized_img

def processImage(self,image,bounding_boxs):

images = []

\# print(image.shape) # 1080,1920,3

labels = []

for index,bbounding_box in bounding_boxs.iterrows():

start_point =

(int(bbounding_box["xmin"]),int(bbounding_box["ymin"]))

end_point =

(int(bbounding_box["xmax"]),int(bbounding_box["ymax"]))

finalImage =

image[start_point[1]:end_point[1],start_point[0]:end_point[0],:]

if self.transform:

finalImage = self.transform(image = finalImage)["image"]

images.append(self.resize_image(finalImage))

labels.append(bounding_boxs.iloc[index])

\# labels.append(bbounding_box["label\_directory"])

return images,labels

def collate_fn(self,data):

_"""_

_label will be in the datafram formate_

_"""_

if self.testValidationFlag:

labels = [j for i in data for j in i["labels"]]

images = [self.resize_image(j) for i in data for j in

i["images"]]

images =

torch.permute(torch.tensor(images,dtype=torch.float),(0,3,1,2))

return images,pd.DataFrame(labels)

else:

labels = [i["labels"] for i in data]

images = [i["image"] for i in data]

images = torch.tensor(images)

torch.permute(images,(0,3,1,2))

df = pd.concat(labels)

df.reset_index(drop = True,inplace = True)

<a name="br17"></a>

return images,df

if \_\_name\_\_ == "\_\_main\_\_":

\# train_df,val_df = readDataFrame("data/aicity2023_track5")

\# print(train_df.shape)

\# print(val_df.shape)

testDataset =

AICITY2023TRACK5TEST("data/aicity2023_track5_test","data/aicity2023_track5_te

st/testImages","data/aicity2023_track5_test/testLabels"

,testValidationFlag=True)

trainLoader = DataLoader(testDataset,batch_size =

20,shuffle=False,collate_fn=testDataset.collate_fn)

print(next(iter(trainLoader))[0].shape)

data = next(iter(trainLoader))[1]

\# print(pd.concat(data,axis = 1))

print(pd.DataFrame(data))

\# testDataset =

AICITY2023TRACK5TEST("data/aicity2023_track5","data/aicity2023_track5/validat

ionImages"

\#

,testValidationFlag = False)

\# print(testDataset[0])

\# transform = A.Compose([

\#

A.HorizontalFlip(p=0.5),

\#

A.RandomBrightnessContrast(p=1),

\# ])

\# trainDataset =

AICITY2023TRACK5("data/aicity2023_track5","data/aicity2023_track5/trainImages

",transform= transform)

\# trainLoader = DataLoader(trainDataset,batch_size = 60,shuffle =

True,collate_fn=trainDataset.collate_fn)

\# print(next(iter(trainLoader))[0].shape)

model.py

import sys

import os

from loader import AICITY2023TRACK5

from torch.utils.data import DataLoader

import torch

import torch.optim as optim

import torchvision

from torchvision import datasets, models, transforms

import torch.nn as nn

from torchsummary import summary

import timm

import tqdm

import albumentations as A

import pandas as pd

import wandb

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix

import numpy as np

<a name="br18"></a>

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIRECTORY = os.path.join(os.getcwd(),"data")

TRAIN_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY,"aicity2023_track5",)

TEST_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY,"aicity2023_track5_test")

VIDEO_TRAIN_DIRECTORY = os.path.join(TRAIN_DATA_DIRECTORY,"videos")

IMAGE_TRAIN_DIRECTORY = os.path.join(TRAIN_DATA_DIRECTORY,"images")

LABEL_TRAIN_DIRECTORY = os.path.join(TRAIN_DATA_DIRECTORY,"labels")

NUMBER_OF_CLASS = 8

LEARNING_RATE = 0.0001

EPOCHS = 1

MODEL_NAME = "efficientnet"

BATCH_SIZE = 1

config={

"learning_rate": LEARNING_RATE,

"epochs": EPOCHS,

"model_name" : MODEL_NAME,

"number_of_class" : NUMBER_OF_CLASS,

"batch_size" : BATCH_SIZE,

}

wandb.login()

class BestModelSaveCallback:

def \_\_init\_\_(self, save_path):

self.save_path = save_path

self.best_accuracy = -1

def \_\_call\_\_(self, accuracy,model):

if accuracy > self.best_accuracy:

self.best_accuracy = accuracy

model.to(device = "cpu")

torch.save(model.state_dict(), self.save_path)

model.to(device=DEVICE)

def set_parameter_requires_grad(model, feature_extracting):

if feature_extracting:

for param in model.parameters():

param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract,

use_pretrained=True):

\# Initialize these variables which will be set in this if statement. Each

of these

\# variables is model specific.

model_ft = None

if model_name == "resnet":

""" Resnet18

"""

model_ft = models.resnet18(pretrained=use_pretrained)

set_parameter_requires_grad(model_ft, feature_extract)

num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs, num_classes)

<a name="br19"></a>

elif model_name == "vgg":

""" VGG11_bn

"""

model_ft = models.vgg11_bn(pretrained=use_pretrained)

set_parameter_requires_grad(model_ft, feature_extract)

num_ftrs = model_ft.classifier[6].in_features

model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

elif model_name == "densenet":

""" Densenet

"""

model_ft = models.densenet121(pretrained=use_pretrained)

set_parameter_requires_grad(model_ft, feature_extract)

num_ftrs = model_ft.classifier.in_features

model_ft.classifier = nn.Linear(num_ftrs, num_classes)

elif model_name == "efficientnet":

"""

Efficient net

"""

model_ft =

timm.create_model('tf_efficientnetv2_s',pretrained=use_pretrained)

set_parameter_requires_grad(model_ft, feature_extract)

num_ftrs = model_ft.classifier.in_features

model_ft.classifier = nn.Sequential(nn.Linear(num_ftrs,num_classes))

elif model_name == "convnext":

"""

Convnext

"""

model_ft =

timm.create_model("convnext_large_384_in22ft1k",pretrained=use_pretrained)

set_parameter_requires_grad(model_ft,feature_extract)

num_ftrs = model_ft.head.fc.in_features

model_ft.head.fc = nn.Linear(num_ftrs,num_classes)

else:

print("Invalid model name, exiting...")

exit()

return model_ft

def train_batch(inputs,model,loss_function,optimizer):

x,y = inputs

x = x.to(DEVICE)

y = y.to(DEVICE)

model.train()

prediction = model(x)

loss = loss_function(prediction,y)

loss.backward()

optimizer.step()

optimizer.zero_grad()

return loss.detach()

def decode(prediction):

<a name="br20"></a>

return torch.argmax(prediction,dim = 1)

@torch.no_grad()

def validation_accuracy_batch(inputs,model):

x,y = inputs

x = x.to(DEVICE)

y = y.to(DEVICE)

model.eval()

prediction = model(x)

#decode function for

class_prediction = decode(prediction)

accuracy = torch.sum(y == class_prediction)

return accuracy.detach().cpu().numpy()

@torch.no_grad()

def validation_loss_batch(inputs,model,loss_function):

x,y = inputs

x = x.to(DEVICE)

y = y.to(DEVICE)

model.eval()

prediction = model(x)

loss = loss_function(prediction, y)

return loss.detach()

def

train(trainLoader,valLoader,model,optimizer,loss_function,epochs,best_model_c

allback):

wandb.watch(model,loss_function,log = "all",log_freq=50)

train_losses =[]

val_losses , val_accuracies = [],[]

for epoch in range(epochs):

train_loss = 0

val_loss = 0

val_accuracy = 0

counter = 0

with tqdm.tqdm(total=len(trainLoader)) as trainingLoop:

for index,batch in enumerate(iter(trainLoader)):

loss = train_batch(batch,model,loss_function,optimizer)

train_loss += loss

counter += 1

trainingLoop.set_description(f"Batch:

{index}/{len(trainLoader)}")

trainingLoop.set_postfix({"training Loss " : loss.item()})

trainingLoop.update(1)

wandb.log({"Training Loss":loss.item() })

train_losses.append(train_loss.item() / len(trainLoader))

<a name="br21"></a>

counter = 0

with tqdm.tqdm(total = len(valLoader)) as validationLoop:

for index,batch in enumerate(iter(valLoader)):

loss = validation_loss_batch(batch,model,loss_function)

val_loss += loss

accuracy = validation_accuracy_batch(batch,model)

val_accuracy += accuracy

counter += batch[1].shape[0]

validationLoop.set_description(f"Batch:

{index}/{len(trainLoader)}")

validationLoop.set_postfix({"Validation Accuracy " :

accuracy.item(),

loss.item()})

"Validation loss " :

wandb.log({"Vlaidation Accuracy" : val_accuracy.item()})

wandb.log({"Validation Loss ": val_loss.item()})

validationLoop.update(1)

best_model_callback(val_accuracy.item()/counter,model)

val_losses.append(val_loss.item() / len(valLoader))

val_accuracies.append(val_accuracy.item()/counter)

return train_losses,val_losses,val_accuracies

def

get_model_optimizer_lossFunction(model_name,featrue_extract,use_pretrained,le

arning_rate):

model = initialize_model(model_name,num_classes =

NUMBER_OF_CLASS,feature_extract=featrue_extract,use_pretrained=use_pretrained

)

model.to(device = DEVICE)

loss_function = nn.CrossEntropyLoss(reduction='mean')

loss_function.to(device=DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,

betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

\# optimizer.to(device = DEVICE)

return model,optimizer,loss_function

def computeConfusionMatrix(model,validationLoader):

model.eval()

ground_truth,predictions = [],[]

for batch in validationLoader:

x,y = batch

x = x.to(DEVICE)

y = y.to(DEVICE)

prediction = model(x)

prediction = decode(prediction)

ground_truth.append(y.detach().cpu().numpy().tolist())

predictions.append(prediction.detach().cpu().numpy().tolist())

return np.array(ground_truth),np.array(predictions)

def Plot(train_losses,val_losses,val_accuracies,model,validationLoader,path):

<a name="br22"></a>

plt.plot(train_losses,label = "train loss")

plt.plot(val_losses,label = "validation loss")

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend()

plt.savefig(os.path.join(path,"Loss.png"))

wandb.log({"Loss": plt})

plt.plot(val_accuracies,label = "validation accuracy")

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.legend()

plt.savefig(os.path.join(path,"Accuracy.png"))

wandb.log({"Accuracy": plt})

ground_truth,predictions = computeConfusionMatrix(model,validationLoader)

cm = confusion_matrix(ground_truth, predictions)

labels =

["motorbike","DHelmet","DNoHelmet","P1Helmet","P1NoHelmet","P2Helmet","P2NoHe

lmet"]

\# Create a heatmap of the confusion matrix

plt.figure(figsize=(15, 15))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,

xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted')

plt.ylabel('True')

plt.title('Confusion Matrix')

plt.savefig(os.path.join(path,"ConfusionMatrix.png"))

wandb.log({"Confusion Matrix": plt})

def main(TRAIN_DATA_DIRECTORY,IMAGE_TRAIN_DIRECTORY,path):

trainDataset =

AICITY2023TRACK5(TRAIN_DATA_DIRECTORY,IMAGE_TRAIN_DIRECTORY)

trainLoader = DataLoader(trainDataset,batch_size =

wandb.config["batch\_size"],shuffle = True,collate_fn=trainDataset.collate_fn)

validationDataset =

AICITY2023TRACK5(TRAIN_DATA_DIRECTORY,IMAGE_TRAIN_DIRECTORY,validation=True)

valLoader = DataLoader(validationDataset,batch_size =

wandb.config["batch\_size"],shuffle =

True,collate_fn=validationDataset.collate_fn)

model,optimizer,loss_function =

get_model_optimizer_lossFunction(wandb.config["model\_name"],False,True,wandb.

config["learning\_rate"])

best_model_callback =

BestModelSaveCallback(save_path=os.path.join(path,'best_model.pth'))

train_losses,val_losses,val_accuracies =

train(trainLoader,valLoader,model,optimizer,loss_function,wandb.config["epoch

s"],best_model_callback)

Plot(train_losses,val_losses,val_accuracies,model,valLoader,path = path)

if \_\_name\_\_ == "\_\_main\_\_":

wandb.init(project="FinalProjectSYDE675",

<a name="br23"></a>

config={

"learning_rate": LEARNING_RATE,

"epochs": EPOCHS,

"model_name" : MODEL_NAME,

"number_of_class" : NUMBER_OF_CLASS,

"batch_size" : BATCH_SIZE,

}

)

main(TRAIN_DATA_DIRECTORY,IMAGE_TRAIN_DIRECTORY,"justTocheck")

ConfusionMatrix.py

import torch

from loader import AICITY2023TRACK5TEST

import matplotlib.pyplot as plt

import cv2

import numpy as np

import os

from torch.utils.data import DataLoader

import tqdm

from model import decode,initialize_model

import pandas as pd

DATA_DIRECTORY = os.path.join(os.getcwd(),"data")

TEST_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY,"aicity2023_track5_test")

TRAIN_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY,"aicity2023_track5",)

VIDEO_TEST_DIRECTORY = os.path.join(TEST_DATA_DIRECTORY, "videos")

IMAGE_TEST_DIRECTORY = os.path.join(TEST_DATA_DIRECTORY, "testImages")

IMAGE_VAL_DIRECTORY = os.path.join(TRAIN_DATA_DIRECTORY,"validationImages")

LABEL_TEST_DIRECTORY = os.path.join(TEST_DATA_DIRECTORY,"testLabels")

LABEL_VAL_DIRECTORY = os.path.join(TRAIN_DATA_DIRECTORY,"valLabels")

\# testDataset =

AICITY2023TRACK5TEST(TEST_DATA_DIRECTORY,IMAGE_TEST_DIRECTORY,LABEL_TEST_DIRE

CTORY,testValidationFlag=True)

\# testLoader = DataLoader(testDataset,batch_size =

20,shuffle=False,collate_fn=testDataset.collate_fn)

\# model = initialize_model("efficientnet",8,False,False)

\# checkpoints = torch.load(os.path.join(os.getcwd(),"src","best_model.pth"))

\# # model.load_state_dict(checkpoints)

\# index = 1

\# data = testDataset[index]

\# images,labels = data["images"],data["labels"]

\# # plt.imshow(images[0])

\# # plt.show()

\# i = 1

\# image_path

<a name="br24"></a>

=os.path.join(IMAGE_TEST_DIRECTORY,labels[i]["label\_directory"].split("/")[-

1].split(".")[0]+".jpeg")

\# fig,(ax1,ax2) = plt.subplots(2,1,figsize = (10,8))

\# image = cv2.imread(image_path)

\# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

\# start_point = (int(labels[i]["xmin"]),int(labels[i]["ymin"]))

\# end_point = (int(labels[i]["xmax"]),int(labels[i]["ymax"]))

\# cv2.rectangle(image,start_point,end_point,color=(0,255,0), thickness=2)

\# final_labels = []

\# softmax = torch.nn.Softmax()

\# with tqdm.tqdm(total = len(testLoader)) as validationLoop:

\#

\#

\#

model.eval()

k = 0

for index,batch in enumerate(iter(testLoader)):

\#

\#

\#

\#

\#

\#

\#

images,labels = batch

prediction = model(images)

class_predictions = decode(prediction)

class_predictions = class_predictions.detach().cpu().numpy()

prediction = softmax(prediction)

labels["class\_predictions\_model"] = class_predictions

labels["confidence"] = torch.max(prediction,dim =

1)[0].detach().numpy()

\#

\#

\#

\#

final_labels.append(labels)

k += 1

if(k > 1):

break

\# df = pd.concat(final_labels)

\# df.reset_index(drop = True,inplace = True)

\# This is for the validation stuff

testDataset =

AICITY2023TRACK5TEST(TRAIN_DATA_DIRECTORY,IMAGE_VAL_DIRECTORY,LABEL_VAL_DIREC

TORY,testValidationFlag=False)

testLoader = DataLoader(testDataset,batch_size =

1,shuffle=False,collate_fn=testDataset.collate_fn)

yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

yolo_model.classes = [0,3]

model = initialize_model("efficientnet",8,True,False)

checkpoints = torch.load(os.path.join(os.getcwd(),"src","best_model.pth"))

model.load_state_dict(checkpoints)

Precision vs Recall .py

import torch

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix

<a name="br25"></a>

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

validationDataset =

AICITY2023TRACK5(TRAIN_DATA_DIRECTORY,IMAGE_VAL_DIRECTORY,validation=True)

valLoader = DataLoader(validationDataset,batch_size = 64,shuffle =

True,collate_fn=validationDataset.collate_fn)

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

\# prediction,\_ = decode(prediction)

prediction = softmax(prediction)

ground_truth.extend(y.detach().cpu().numpy().tolist())

predictions.extend(prediction.detach().cpu().numpy().tolist())

return np.array(ground_truth),np.array(predictions)

ground_truth, predictions = computePredictions(model,valLoader)

print(predictions.shape)

precision = {}

recall = {}

categories = {0 : "motorbike",1 : "DHelmet",2 : "DNoHelmet",3 : "P1Helmet",4

: "P1NoHelmet",5 : "P2Helmet",6: "P2NoHelmet", 7 : "Other"}

n_classes = 8 # number of classes in your multi-class problem

for i in range(n_classes):

precision[i], recall[i], \_ = precision_recall_curve(ground_truth == i,

predictions[:,i])

<a name="br26"></a>

plt.figure(figsize=(8, 6))

for i in range(n_classes):

plt.plot(recall[i], precision[i], lw=2, label=f'Class {categories[i]}')

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.legend()

plt.title('Precision-Recall Curve for Multi-Class Classification')

plt.show()

The folder structure should be as follows

All the notebooks should go into notebooks folder and all the python script should go into src

folder.

<a name="br27"></a>

Helmet No-Helmet with SVM

As the number of cropped images was humongous I was unable to run this program for full

dataset

Code

<a name="br28"></a>

Conclusion

The results can be improved by using diﬀerent data augmentaꢀon techniques and creaꢀng

custom dataset as menꢀoned above with few modiﬁcaꢀon. Also diﬀerent model can be tried

with to check which model best results in beꢁer accuracy

Overall It seems that with just the yolo model the model is not able to detect all the object

present in the image whereas with the second approach of using yolo and other classiﬁer it

seems as it’s detecꢀng too much so may be some ensemble method might also help improve

the accuracy.
