# How to Yolo
___


The main information on setting and running YOLO came from this [site](https://github.com/experiencor/basic-yolo-keras
) https://github.com/experiencor/basic-yolo-keras


## Setup

Copy the content of the folder [./yolo](./yolo) locally. 

Be sure to copy the initial training weights at 
\\metrix-fsrv-07\WORKING_DATA\Data\Job Files - Current\1710-01 MosquitoNET\02 - Engineering\03 - Classifier Development\YOLO\full_yolo_backend.h5 and trained weights at 
\\metrix-fsrv-07\WORKING_DATA\Data\Job Files - Current\1710-01 MosquitoNET\02 - Engineering\03 - Classifier Development\YOLO\mos_fm.h5 from the Kinemetrix server

## Config

### Configuration File

The configuration file describing the training architecture, folders for input images and image annotations is located in the [conf_mf.json](./yolo/conf_mf.json) file.

In our case the labels consist of the following 

	"labels":               ["m","f","j"]

The "stats" section of the configuration file is used for creating confusion matrix when estimating accuracy on the test set. 

On Linux box, the location of the training images and annotations is described in the "train" section: 

	"train_image_folder":   "/home/user/assignment1/mosquito/yolo/data/train_img/",
    "train_annot_folder":   "/home/user/assignment1/mosquito/yolo/data/train_ann/",

Please note for training YOLO on windows the path for training folders looks the following: 

    "train_image_folder":   "C:/_project/1710-01 MosquitoNet/App/AI/yolo/data/img/",
    "train_annot_folder":   "C:/_project/1710-01 MosquitoNet/App/AI/yolo/data/ann/", 

## Train

Under Linux run the bash script trainNet.sh 

<p align="center">
  <img src=".\Images\run_train_script.jpg" width="500"/>  
</p>


Under windows run command prompt, start a conda environment with Tensorflow and Keras, naviate to where the training scripts for yolo are located and run the following command
	
	python3 train.py -c conf_mf.json


## Run and Verify

After the training  the will be stored in file  'mos_fm.h5' ( it is indicated in the config.json setting)
	
 	"saved_weights_name":   "mos_mf.h5",

One way to run the test is to probe the trained network on a single image: 
	
on a Linux run

	 > $ ./prdict_img.sh
	
on windows

	python3 predict.py  -c conf_mf.json -w mos_mf.h5 -i ./data/img/051_female.jpg

The script will generate a processed image in the same folder as the input image with the suffix "detected"
<p align="center">
  <img src=".\Images\img_single_proc.jpg"  width="500"/>  
</p>

The detected class will be displayed like this 

<p align="center">
  <img src=".\Images\img_processed.jpg" height = "500" width="500"/>  
</p>


To test the trained network in a more rigorous manner, we can run the classifier against a  collection of images. For example, images with known labels are in folder. Every images carries a class label in its name ( "__male", " __female", "__junk")  

<p align="center">
  <img src=".\Images\img_labeled.jpg" height = "500" width="500"/>  
</p>



Execute the following command to run the classifier on a folder containing test images

	 python3 predictfld-stat.py -c conf_mf.json -w mos_mf.h5 -i ./data/test_mf


When the classifier runs, the debug print displays results similar to the following:


<p align="center">
  <img src=".\Images\classifier_debug.jpg"  width="500"/>  
</p>

Also, at the end of the execution the script produces a confusion matrix:

<p align="center">
  <img src=".\Images\confusion_matrix.jpg"  width="500"/>  
</p>

Where the rows indicate the classes of the files being tested, the columns indicate the classes assigned to the files. In case if the classifier could not assign a class, i.e., the highest result of the classification result was below the threshold, the image would be assigned under "unknown" category. 


The log file for the prediction process will be stored at the "image folder"/logs/ subfolder:


<p align="center">
  <img src=".\Images\img_log_files.jpg"  width="500"/>  
</p>