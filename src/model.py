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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIRECTORY = os.path.join(os.getcwd(),"data")
TRAIN_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY,"aicity2023_track5",)
TEST_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY,"aicity2023_track5_test")
VIDEO_TRAIN_DIRECTORY = os.path.join(TRAIN_DATA_DIRECTORY,"videos")
IMAGE_TRAIN_DIRECTORY = os.path.join(TRAIN_DATA_DIRECTORY,"images")
LABEL_TRAIN_DIRECTORY = os.path.join(TRAIN_DATA_DIRECTORY,"labels")
NUMBER_OF_CLASS  = 8
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
    def __init__(self, save_path):
        self.save_path = save_path
        self.best_accuracy = -1

    def __call__(self, accuracy,model):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            model.to(device = "cpu")
            torch.save(model.state_dict(), self.save_path)
            model.to(device=DEVICE)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

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
        model_ft = timm.create_model('tf_efficientnetv2_s',pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Sequential(nn.Linear(num_ftrs,num_classes))
        
    elif model_name == "convnext":
        """
        Convnext
        """
        model_ft = timm.create_model("convnext_large_384_in22ft1k",pretrained=use_pretrained)
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

def train(trainLoader,valLoader,model,optimizer,loss_function,epochs,best_model_callback):
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
                trainingLoop.set_description(f"Batch: {index}/{len(trainLoader)}")
                trainingLoop.set_postfix({"training Loss " : loss.item()})
                trainingLoop.update(1)
                wandb.log({"Training Loss":loss.item() })
            
        train_losses.append(train_loss.item() / len(trainLoader))
        
        counter = 0  

        with tqdm.tqdm(total = len(valLoader)) as validationLoop:
            for index,batch in enumerate(iter(valLoader)):
    
                loss = validation_loss_batch(batch,model,loss_function)
                val_loss += loss
                accuracy = validation_accuracy_batch(batch,model)
                val_accuracy += accuracy
                counter += batch[1].shape[0]
                validationLoop.set_description(f"Batch: {index}/{len(trainLoader)}")
                validationLoop.set_postfix({"Validation Accuracy " : accuracy.item(),
                                            "Validation loss " : loss.item()}) 
                wandb.log({"Vlaidation Accuracy" : val_accuracy.item()})
                wandb.log({"Validation Loss ": val_loss.item()})
                validationLoop.update(1)
        
        best_model_callback(val_accuracy.item()/counter,model)
        
        val_losses.append(val_loss.item() / len(valLoader))
        val_accuracies.append(val_accuracy.item()/counter)
    return train_losses,val_losses,val_accuracies


def get_model_optimizer_lossFunction(model_name,featrue_extract,use_pretrained,learning_rate):
    model = initialize_model(model_name,num_classes = NUMBER_OF_CLASS,feature_extract=featrue_extract,use_pretrained=use_pretrained)
    model.to(device = DEVICE)

    loss_function = nn.CrossEntropyLoss(reduction='mean')
    loss_function.to(device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # optimizer.to(device = DEVICE)
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
    labels = ["motorbike","DHelmet","DNoHelmet","P1Helmet","P1NoHelmet","P2Helmet","P2NoHelmet"]

    # Create a heatmap of the confusion matrix
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(path,"ConfusionMatrix.png"))
    wandb.log({"Confusion Matrix": plt})


def main(TRAIN_DATA_DIRECTORY,IMAGE_TRAIN_DIRECTORY,path):
    trainDataset = AICITY2023TRACK5(TRAIN_DATA_DIRECTORY,IMAGE_TRAIN_DIRECTORY)
    trainLoader = DataLoader(trainDataset,batch_size = wandb.config["batch_size"],shuffle = True,collate_fn=trainDataset.collate_fn)

    validationDataset = AICITY2023TRACK5(TRAIN_DATA_DIRECTORY,IMAGE_TRAIN_DIRECTORY,validation=True)
    valLoader = DataLoader(validationDataset,batch_size = wandb.config["batch_size"],shuffle = True,collate_fn=validationDataset.collate_fn)

    model,optimizer,loss_function = get_model_optimizer_lossFunction(wandb.config["model_name"],False,True,wandb.config["learning_rate"])
    best_model_callback = BestModelSaveCallback(save_path=os.path.join(path,'best_model.pth'))
    train_losses,val_losses,val_accuracies = train(trainLoader,valLoader,model,optimizer,loss_function,wandb.config["epochs"],best_model_callback)
    Plot(train_losses,val_losses,val_accuracies,model,valLoader,path = path)


if __name__ == "__main__":
    wandb.init(project="FinalProjectSYDE675",
    config={
 
       "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "model_name" : MODEL_NAME,
        "number_of_class" : NUMBER_OF_CLASS,
        "batch_size" : BATCH_SIZE,
    }
    )
    main(TRAIN_DATA_DIRECTORY,IMAGE_TRAIN_DIRECTORY,"justTocheck")