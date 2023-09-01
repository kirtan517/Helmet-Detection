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

# testDataset = AICITY2023TRACK5TEST(TEST_DATA_DIRECTORY,IMAGE_TEST_DIRECTORY,LABEL_TEST_DIRECTORY,testValidationFlag=True)
# testLoader = DataLoader(testDataset,batch_size = 20,shuffle=False,collate_fn=testDataset.collate_fn)

# model = initialize_model("efficientnet",8,False,False)
# checkpoints = torch.load(os.path.join(os.getcwd(),"src","best_model.pth"))
# # model.load_state_dict(checkpoints)

# index = 1

# data = testDataset[index]
# images,labels = data["images"],data["labels"]
# # plt.imshow(images[0])
# # plt.show()
# i = 1

# image_path =os.path.join(IMAGE_TEST_DIRECTORY,labels[i]["label_directory"].split("/")[-1].split(".")[0]+".jpeg")
# fig,(ax1,ax2) = plt.subplots(2,1,figsize = (10,8))
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# start_point = (int(labels[i]["xmin"]),int(labels[i]["ymin"]))
# end_point = (int(labels[i]["xmax"]),int(labels[i]["ymax"]))
# cv2.rectangle(image,start_point,end_point,color=(0,255,0), thickness=2)

# final_labels = []
# softmax = torch.nn.Softmax()

# with tqdm.tqdm(total = len(testLoader)) as validationLoop:
#     model.eval()
#     k = 0
#     for index,batch in enumerate(iter(testLoader)):

#         images,labels = batch
#         prediction = model(images)
#         class_predictions = decode(prediction)
#         class_predictions = class_predictions.detach().cpu().numpy()
#         prediction = softmax(prediction)
#         labels["class_predictions_model"] = class_predictions
#         labels["confidence"] = torch.max(prediction,dim = 1)[0].detach().numpy()
#         final_labels.append(labels)
#         k += 1
#         if(k > 1):
#             break

# df = pd.concat(final_labels)
# df.reset_index(drop = True,inplace = True)

# This is for the validation stuff 
testDataset = AICITY2023TRACK5TEST(TRAIN_DATA_DIRECTORY,IMAGE_VAL_DIRECTORY,LABEL_VAL_DIRECTORY,testValidationFlag=False)
testLoader = DataLoader(testDataset,batch_size = 1,shuffle=False,collate_fn=testDataset.collate_fn)

yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
yolo_model.classes = [0,3]


model = initialize_model("efficientnet",8,True,False)
checkpoints = torch.load(os.path.join(os.getcwd(),"src","best_model.pth"))
model.load_state_dict(checkpoints)

