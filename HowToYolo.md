# How to Yolo
___


The main information on setting and running YOLO came from this [site](https://github.com/experiencor/basic-yolo-keras
) https://github.com/experiencor/basic-yolo-keras


## Setup

Copy the content of the folder [./yolo](./yolo) locally. 

Be sure to copy the initial [training weights](\\metrix-fsrv-07\WORKING_DATA\Data\Job Files - Current\1710-01 MosquitoNET\02 - Engineering\03 - Classifier Development\YOLO\full_yolo_backend.h5) and [trained weights](\\metrix-fsrv-07\WORKING_DATA\Data\Job Files - Current\1710-01 MosquitoNET\02 - Engineering\03 - Classifier Development\YOLO\mos_fm.h5) from the Kinemetrix server

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



## Run and Verify
