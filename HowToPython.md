# How-Tos for Deep Nets and Python 


1. [Generate altered images](./python/genImages.py) from a collection of images using Keras's ImageGenerator 

2. [Generate directory](./python/genTrainDir.py) with training images. In this case the 'xml' file is generated on the fly using xmlTemlate.

3. [Generate directory](./python/genTrainDirFromSeparateDirs.py) with training images split in directories. In this case the 'xml' file is generated on the fly using xmlTemlate.
To run the script follow the command below:

		python3 genTrainDirFromSeparateDirs.py -i /home/hakim/assignment1/mosquito/yolo/data/train_set_03/ -o /home/hakim/assignment1/mosquito/yolo/data/train_set_04

The structure of the input directory is as follows:
 
		input_dir  \
	               +--\images_f
				   |
				   +--\images_m
				   |
				   +--\images_j

All images are same size, naming of the images does not matter. What is important what folder images are assigned in . 

<p align="center">
  <img src=".\Images\genImagesFromDirs.jpg"  width="500"/>  
</p>
The expected output directory:
 
	input_output\
               +--\img
			   |
			   +--\ann

Where .\img contains images and .\ann contains XML files

<p align="center">
  <img src=".\Images\genImagesFromDirsOutput.jpg"  width="500"/>  
</p>