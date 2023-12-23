## YOLO v5 Model

In this approach, all the images and labels were trained on the yolov5m model.

<br>

<p align="center">
  <img src="https://github.com/kirtan517/Helmet-Detection/blob/main/Images/confusion_matrix.png" alt="Confusion Matrix" width="800">
</p>

As it can be seen from the correlation matrix that the model was unable to predict the P1Helmet correctly at all of the predictions for that class were wrong and same goes for P2Helmet and P2NoHelmet as well.

<p align="center">
  <img src="https://github.com/kirtan517/Helmet-Detection/blob/main/Images/labels.jpg" alt="Confusion Matrix" width="800">
</p>

As it can be seen from the above figure the there is high class mismatch as there are small number of bounding box which are being identified as the P1Helmet , P2Helmet and P2NoHelmet this might be causing the model to perform poorly on the P1Helmet classification.

<p align="center">
  <img src="https://github.com/kirtan517/Helmet-Detection/blob/main/Images/F1_curve.png" alt="P Curve" width="400">
  <img src="https://github.com/kirtan517/Helmet-Detection/blob/main/Images/PR_curve.png" alt="R Curve" width="400">
</p>

The major for such a low prediction for P2 might be that having 3 people on the motorbike is a rare event and further to classify that it’s the p2 model need to first identify that there is p1 present in the motorbike so such sort of sequential learning might be causing the model to predict poorly on this particular class.

<p align="center">
  <img src="https://github.com/kirtan517/Helmet-Detection/blob/main/Images/Train_test_yolov5.png" alt="Confusion Matrix" width="800">
</p>

<p align="center">
  <img src="https://github.com/kirtan517/Helmet-Detection/blob/main/Images/P_curve.png" alt="P Curve" width="400">
  <img src="https://github.com/kirtan517/Helmet-Detection/blob/main/Images/R_curve.png" alt="R Curve" width="400">
</p>

