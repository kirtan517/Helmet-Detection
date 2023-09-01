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

def generateDataFrame(trainDataDirectory):
    column_names = ["video_id", "frame", "track_id", "bb_left", "bb_top", "bb_width","bb_height","category"]
    df = pd.read_csv(os.path.join(trainDataDirectory,"gt.txt") , names = column_names)
    df["category"] = df["category"] - 1
    df["image_path"] = df["video_id"].apply(lambda x : f"{x:03}")+ "_"+ df["frame"].apply(lambda x : f"{x-1:03}") + ".jpeg"
    df["normalized_center_x"] = (df["bb_left"] + df["bb_width"]//2) / 1920
    df["normalized_center_y"] = (df["bb_top"] + df["bb_height"]//2) / 1080
    df["normalized_width"] = df["bb_width"] / 1920
    df["normalized_height"] = df["bb_height"] / 1080
    df = df[~df["image_path"].isin(["034_200.jpeg","034_201.jpeg","034_202.jpeg","034_203.jpeg","034_204.jpeg"])].reset_index()
    split_on = df["image_path"].unique()
    train_split,val_split = train_test_split(split_on,test_size=0.10,shuffle=True)
    train_df = df[df["image_path"].isin(train_split)]
    val_df = df[df["image_path"].isin(val_split)]
    train_df = train_df.reset_index()
    val_df = val_df.reset_index()
    train_df.to_csv(os.path.join(trainDataDirectory,"train.csv"))
    val_df.to_csv(os.path.join(trainDataDirectory,"val.csv"))
    return train_df,val_df

def readDataFrame(trainDataDirectory):
    if(os.path.isfile(os.path.join(trainDataDirectory,"train.csv")) and os.path.isfile(os.path.join(trainDataDirectory,"val.csv"))):
        train_df = pd.read_csv(os.path.join(trainDataDirectory,"train.csv"))
        val_df = pd.read_csv(os.path.join(trainDataDirectory,"val.csv"))
    else:
        train_df,val_df = generateDataFrame(trainDataDirectory)
    return train_df,val_df

class AICITY2023TRACK5(Dataset):
    def __init__(self,Training_directory,Training_image_directory,resized_width=100,resized_height=100,transform = None,validation = False):
        """
        path -> Training Directory
        """
        self.Training_directory = Training_directory
        self.Training_image_directory = Training_image_directory
        self.validation = validation
        if validation:
            _ , self.df = readDataFrame(Training_directory)
        else:
            self.df, _ = readDataFrame(Training_directory)
        self.unique_images = self.df["image_path"].unique()
        self.labelsToClasses  = {0 : "motorbike",1 : "DHelmet",2 : "DNoHelmet",3 : "P1Helmet",4 : "P1NoHelmet",5 :  "P2Helmet",6: "P2NoHelmet"}
        self.resized_width = resized_width
        self.resized_height = resized_height
        self.transform = transform
        
    def __len__(self):
        return len(self.unique_images)
    
    def __getitem__(self,index):
        image_name = self.unique_images[index]
        image = cv2.imread(os.path.join(self.Training_image_directory,image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images,labels = self.processImage(image,self.df[self.df["image_path"] == image_name])


        return {"images": images,"labels":labels,"image":image}
    
    def processImage(self,image,bounding_boxs):
        
        images = []
        labels = []
        # print(image.shape) # 1080,1920,3
        
        for index,bbounding_box in bounding_boxs.iterrows():
            start_point = (bbounding_box["bb_left"],bbounding_box["bb_top"])
            end_point = (bbounding_box["bb_width"]+start_point[0],bbounding_box["bb_height"]+start_point[1])
            
            finalImage = image[start_point[1]:end_point[1],start_point[0]:end_point[0],:]

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
        images = torch.permute(torch.tensor(images,dtype=torch.float),(0,3,1,2)) # Batch x channels x height x width
        labels = torch.tensor(labels,dtype=torch.long)
        return images , labels
    
class AICITY2023TRACK5TEST(Dataset):

    def __init__(self,test_directory,test_image_directory,test_label_directory,testValidationFlag = False,transform = None,resized_width = 100,resized_height=100):
        """
        testValidationFlag -> true when the dataset is for test 
        """
        self.directory = test_directory
        self.test_image_directory = test_image_directory
        self.testValidationFlag = testValidationFlag
        self.test_label_directory = test_label_directory
        self.resized_width = resized_width
        self.resized_height = resized_height
        if testValidationFlag:
            images = os.listdir(test_image_directory)
            images_dir = [os.path.join(test_image_directory,i) for i in images]
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
                      "labels" : labelFinal,"label_directory" : labelDirectory,}

            self.df = pd.DataFrame(output)
        else:
            file_path = os.path.join(test_directory,"val.csv")
            if not os.path.isfile(file_path):
                _,self.df = readDataFrame(test_directory)
            else:
                self.df = pd.read_csv(file_path)
            self.df["images"] = self.df["image_path"].apply(lambda x : x.split(".")[0])
            self.df["image_directory"] = self.df["image_path"].apply(lambda x : os.path.join(test_image_directory,x))

        self.transform = transform 
        self.uniques = self.df["image_directory"].unique()

    def __len__(self):
        return len(self.uniques)
    
    def __getitem__(self,index):
        self.image_path = self.uniques[index]
        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.testValidationFlag:
            # This is the test thing
            label = self.df[self.df["image_directory"] == self.image_path]
            labels = pd.read_csv(label["label_directory"].item())
            #labels should be in the list formate
            labels["label_directory"] = label["label_directory"].item()
            images,labels = self.processImage(image,labels)  
            return {"images":images,"labels":labels}
        else:
            self.labels = self.df[self.df["image_directory"] == self.image_path]
            return {"image":image,"labels":self.labels}
        
    def resize_image(self,image):
        new_size = (self.resized_width, self.resized_height)
        resized_img = cv2.resize(image, new_size)
        return resized_img
        
    def processImage(self,image,bounding_boxs):
        
        images = []
        # print(image.shape) # 1080,1920,3
        labels = []

        for index,bbounding_box in bounding_boxs.iterrows():
            start_point = (int(bbounding_box["xmin"]),int(bbounding_box["ymin"]))
            end_point = (int(bbounding_box["xmax"]),int(bbounding_box["ymax"]))
            
            finalImage = image[start_point[1]:end_point[1],start_point[0]:end_point[0],:]

            if self.transform:
                finalImage = self.transform(image = finalImage)["image"]


            images.append(self.resize_image(finalImage))
            labels.append(bounding_boxs.iloc[index])
            # labels.append(bbounding_box["label_directory"])
            
        return images,labels
     
    def collate_fn(self,data):
        """
        label will be in the datafram formate
        """
        
        if self.testValidationFlag:
            labels = [j for i in data for j in i["labels"]] 
            images = [self.resize_image(j) for i in data for j in i["images"]]
            images = torch.permute(torch.tensor(images,dtype=torch.float),(0,3,1,2))
            return images,pd.DataFrame(labels)
        else:
            labels = [i["labels"] for i in data] 
            images = [i["image"] for i in data]
            images = torch.tensor(images)
            torch.permute(images,(0,3,1,2))
            df = pd.concat(labels)
            df.reset_index(drop = True,inplace = True)
            return images,df
        
if __name__ == "__main__":
    # train_df,val_df = readDataFrame("data/aicity2023_track5")
    # print(train_df.shape)
    # print(val_df.shape)
    testDataset = AICITY2023TRACK5TEST("data/aicity2023_track5_test","data/aicity2023_track5_test/testImages","data/aicity2023_track5_test/testLabels"
                                       ,testValidationFlag=True)
    trainLoader = DataLoader(testDataset,batch_size = 20,shuffle=False,collate_fn=testDataset.collate_fn)
    print(next(iter(trainLoader))[0].shape)
    data = next(iter(trainLoader))[1]
    # print(pd.concat(data,axis = 1))
    print(pd.DataFrame(data))

    # testDataset = AICITY2023TRACK5TEST("data/aicity2023_track5","data/aicity2023_track5/validationImages"
    #                                    ,testValidationFlag = False)
    # print(testDataset[0])
    # transform = A.Compose([
    #     A.HorizontalFlip(p=0.5),
    #     A.RandomBrightnessContrast(p=1),
    # ])
    # trainDataset = AICITY2023TRACK5("data/aicity2023_track5","data/aicity2023_track5/trainImages",transform= transform)
    # trainLoader = DataLoader(trainDataset,batch_size = 60,shuffle = True,collate_fn=trainDataset.collate_fn)
    # print(next(iter(trainLoader))[0].shape)