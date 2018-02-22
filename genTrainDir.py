#from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
#from keras import backend as K
import os
import numpy as np
from PIL import Image
import sys
import cv2
from shutil import copyfile
from mako.template import Template
from mako.runtime import Context

from io import StringIO

#from keras.datasets import mnist






def genXMLContent(  _img_filename, _clsLbl):


	 
	xmlTempate = Template(filename='xmlTemplate.txt')
	buf = StringIO()
	#cxt   = Context(buf, folder_name ='imagesAll', img_filename = 'm0001.jpg', img_filepath = 'imagesAll/m0001.jpg', clsLbl='m', xmin='100', ymin='101')
	cxt   = Context(buf, folder_name   = 'imagesAll',
		                 img_filename  = _img_filename, 
		                 img_filepath = 'imagesAll/' + _img_filename, 
		                 clsLbl=_clsLbl, 
		                 xmin='200', 
		                 ymin='200',
		                 xmax='900', 
		                 ymax='900'
		                 )
	xmlTempate.render_context(cxt)
	xmlContent = buf.getvalue()
	return xmlContent





print ('Cobmine images for YOLO Training')
imagePath     = '../data/img'
outputImgPath = '../data/train_img'
outputAnnPath = '../data/train_ann'


if os.path.isdir(outputImgPath)==False:
		os.makedirs(outputImgPath)

if os.path.isdir(outputAnnPath)==False:
		os.makedirs(outputAnnPath)



file_names = []
classes = ['m', 'f', 'j']
for clsLbl in classes:
	
	

	imagePath = 'images_' + clsLbl
	
	print (' ----- - - - - - - - - - - - ' , imagePath)
	
	file_names = [fn for fn in os.listdir(imagePath)]

	fileID = 1 

	for f in file_names:
		newFileName = clsLbl +  str(fileID).zfill(4)
		print (' {} -> {} '.format(f, newFileName))

		# COPY IMAGE FILE
		srcFileName = os.path.join(imagePath,f)
		dstFileName = os.path.join(outputImgPath,newFileName +'.jpg')		
		copyfile(srcFileName, dstFileName)

		# CREATE XML FILE

		xmlContent = genXMLContent(newFileName + '.jpg', clsLbl)
		xmlFileName = os.path.join(outputAnnPath,newFileName +'.xml')

		# SAVE XML FILE

		xmlFile = open(xmlFileName, "w")
		xmlFile.write	(xmlContent)
		xmlFile.close()








		fileID +=1 
		#if fileID > 20:
		#	break



sys.exit()

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
file_names = [fn for fn in os.listdir(imagePath)]

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
	
	
	for ii, f in enumerate(filesToAugment):

		#print (labels[ii], f)
		fname = os.path.join(imagePath,f)

		
		
		n  += 1
				
		if (n%1==0):
			print ('.', end='')
			sys.stdout.flush()	

	
	

	

	
	
	if os.path.isdir(outputImagePath)==False:
		os.makedirs(outputImagePath)

	i = 0 
	print ('')
	print (' = = = =')
	print ("x train shape: ", x_train.shape)
	print (' = = = =')
	print ('')

	


	

