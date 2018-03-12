# How to Use Tensorboard to view the training process
___




## Start Tensorboard

The yolo training process create a log file in in ~/logs/ directory. Locate the "latest" yolo log folder. As it shown in the image below, it is "yolo_18". 

<p align="center">
  <img src=".\Images\tensorboard_01.jpg" width="500"/>  
</p>

In a terminal window run the following command: 
	
	tensorboard --logdir ~/logs/yolo_18

The tensorboard server starts on the http://localhost:6006. In this case the name of the station is "DeepNetStation01", hence the link to tensorboard interface is http://

<p align="center">
  <img src=".\Images\tensorboard_02.jpg" width="500"/>  
</p>

## View Tensorboard interface

In a web browser open the link  http://localhost:6006. 

The Tensorboard graphs looks as the following:

<p align="center">
  <img src=".\Images\tensorboard_03.jpg" width="500"/>  
</p>

