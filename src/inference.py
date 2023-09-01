import torch
from ultralytics import YOLO
from loader import AICITY2023TRACK5TEST
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader 
import wandb

DATA_DIRECTORY = os.path.join(os.getcwd(),"data")
TEST_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY,"aicity2023_track5_test")
TRAIN_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY,"aicity2023_track5",)

VIDEO_TEST_DIRECTORY = os.path.join(TEST_DATA_DIRECTORY, "videos")

IMAGE_VAL_DIRECTORY = os.path.join(TRAIN_DATA_DIRECTORY,"validationImages")
IMAGE_TEST_DIRECTORY = os.path.join(TEST_DATA_DIRECTORY, "testImages")

LABEL_VAL_DIRECTORY = os.path.join(TRAIN_DATA_DIRECTORY,"valLabels")


def test(testLoader,yoloModel):
    images = next(iter(testLoader))
    prediction = yoloModel(images)
    print(prediction)


def main():
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=1),
    ])
    testDataset = AICITY2023TRACK5TEST(TEST_DATA_DIRECTORY,IMAGE_TEST_DIRECTORY,testValidationFlag=True)
    testLoader = DataLoader(testDataset,batch_size=32,shuffle=True)
    images = next(iter(testLoader))

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s',pretrained=True)
    model.train()

    print(images.shape)

if __name__ == "__main__":
    main()