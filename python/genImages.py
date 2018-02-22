'''

	genTrainDir - a script to generate images augmented from  
	a source directory. The augmentation takes place using 
	ImageDataGenerator from keras package.
	This particular script was designed to augments images from the
	mosquito project. The mosquito training files come in three categories
	male, female, junk, i.e., (m, f, j).
	to run execute : python -c [m,f,j] genImages.py 
	tree structure relative to the script's location
	/genImages.py
	|
	/images_m/*.jpg  (images with male mosquitos)
	|
	/images_f/*.jpg  (images with female mosquitos)
	|
	/images_j/*.jpg  (images with junk mosquitos)
	|
	/images/*        (augmented images)


'''

from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras import backend as K
import os
import numpy as np
from PIL import Image
import sys
import cv2


#from keras.datasets import mnist


#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#print ("x train: ", X_train.shape)
#print ("y train: ", y_train.shape)


print ('Generage  images')
imagePath = '../data/img'
outputImagePath = 'images_'


if __name__ == '__main__':
	if (len(sys.argv)< 3):
		print ('need to specify class parameter: (m, f, j)')
		print ('python -c [m,f,j] genImages.py ')
		sys.exit()
	elif (sys.argv[1] != '-c'):
		print ('need to specify class parameter "-c" ')
		print ('python -c [m,f,j] genImages.py ')
		sys.exit()
	elif (sys.argv[2] not in ['m', 'f', 'j']):
		print ('need to specify allowed class parameter:  m, f, or j')
		print ('python -c [m,f,j] genImages.py ')
		sys.exit()

imagePathSuffix = sys.argv[2]

classFilename = ''
if (imagePathSuffix == 'm' ):
	classFilename = '_male'
	labelId =1
elif (imagePathSuffix == 'f' ):
	classFilename = '_female'
	labelId = 2
elif (imagePathSuffix == 'j' ):
	classFilename = '_junk'
	labelId = 3

outputImagePath = outputImagePath + imagePathSuffix
print ('ouput image path : ',outputImagePath)




included_extensions = ['jpg', 'bmp', 'png', 'gif']
file_names = [fn for fn in os.listdir(imagePath)
              if fn.find(classFilename)!=-1 and any(fn.endswith(ext) for ext in included_extensions)]

labels = [labelId for f in file_names]


totalFiles = len (file_names)




miniBatchSize = 20
miniBatchCount = int(totalFiles /miniBatchSize)
shift = 0.01

print ('miniBatchSize*miniBatchCount:  ', miniBatchSize*miniBatchCount)
print ('miniBatchSize               :  ', miniBatchSize)

for i in range(0,miniBatchSize*miniBatchCount, miniBatchSize) :
	print ('i {}: from {} to {}'.format(i,i, i+miniBatchSize ))
	filesToAugment = file_names[i:i+miniBatchSize]
	
	x =np.empty((0, 1200, 1200, 3))
	y = []
	n = 0 
	images= []
	for ii, f in enumerate(filesToAugment):

		#print (labels[ii], f)
		fname = os.path.join(imagePath,f)

		image = cv2.imread(fname)
		images.append(image)

		im = Image.open(fname)
		imArray = np.array(im)

		x = np.append(x,[imArray], axis =0)
		y = np.append(y, labelId)
		n  += 1
				
		if (n%1==0):
			print ('.', end='')
			sys.stdout.flush()	

	x_train = np.array(images)
	x_train = x_train.astype('float32')

	y_train = y

	#fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}. 
	# constant leaves black corners on rotation
	# nearest = drags pixel values to corners
	# wrap is the best for our application

	datagen = ImageDataGenerator(fill_mode = 'wrap', rotation_range=90, horizontal_flip=True, vertical_flip=True, width_shift_range=shift, height_shift_range=shift)
	#datagen = ImageDataGenerator(zca_whitening=True)
	if os.path.isdir(outputImagePath)==False:
		os.makedirs(outputImagePath)

	i = 0 
	print ('')
	print (' = = = =')
	print ("x train shape: ", x_train.shape)
	print (' = = = =')
	print ('')

	for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=10, save_to_dir=outputImagePath, save_prefix='aug', save_format='jpg'):
		i += 1
		print ('>', end='')
		sys.stdout.flush()
		if i> 10:
			break;

	print ('')


	

