#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json
from fnmatch import fnmatch
from shutil import copyfile
import datetime

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]=""

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-i',
    '--input',
    help='path to input folder')

argparser.add_argument(
    '-o',
    '--output',
    help='path to output folder')



def _main_(args):
 
    input_path  = args.input
    output_path = args.output

    print ('input  path : {}'.format(input_path))
    print ('output path : {}'.format(output_path))

    pattern  = '*.jpg'

    imageIndex  = 0 
    k = 1
    for path, subdirs, files in os.walk(input_path):
        for name in files:
            if fnmatch(name, pattern):
                filename = os.path.join(path, name)
                newname  = os.path.join(output_path, str(imageIndex).zfill(4)) + ".jpg"


                #imageIndex = imageIndex +  1 
                #print ("{}  - - {}".format(filename, str(imageIndex).zfill(4)))
                print ("{} - - - {index:04d} -- {newfilename}".format(filename, index = imageIndex, newfilename=newname))
                copyfile(filename, newname)
                imageIndex = imageIndex + 1
                
                #imageIndex = imageIndex + 1

    			#image = cv2.imread(filename)
    			

    				        		
    			#cv2.imwrite(os.path.join(path, "detected_" + name), image)
    			#log_str += '\n'

    
if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
