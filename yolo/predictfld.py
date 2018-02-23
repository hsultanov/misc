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
import datetime

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]=""

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    help='path to images')

def _main_(args):
 
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(architecture        = config['model']['architecture'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################    

    print (weights_path)
    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################

    pattern  = '*.jpg'
    log_file = 'log_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.txt'
    log_str  = ''

    for path, subdirs, files in os.walk(image_path):
    	for name in files:
    		if fnmatch(name, pattern):
    			filename = os.path.join(path, name)
    			print (filename)
    			log_str += path + ',' + name

    			image = cv2.imread(filename)
    			boxes = yolo.predict(image)
    			image = draw_boxes(image, boxes, config['model']['labels'])

    			print (len(boxes), 'boxes are found')
    			for box in boxes:
    				box_label = config['model']['labels'][box.get_label()]
    				box_score = box.get_score()
    				log_str += ',' + box_label + ',' + str(box_score)

    				        		
    			cv2.imwrite(os.path.join(path, "detected_" + name), image)
    			log_str += '\n'

    text_file = open(log_file, "w")
    text_file.write(log_str)
    text_file.close()

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
