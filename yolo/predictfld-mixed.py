#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes, BoundBox
from frontend import YOLO
import json
from fnmatch import fnmatch
import datetime
import collections

from time import sleep
import sys


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

    aDict = config['stats']['class_labels']
    classes_to_eval = list(aDict.keys())       
    classes_stat = classes_to_eval + ["unknown"]
	
    aDictClassID = {}
    cid = 0
    for c in classes_to_eval:
        aDictClassID.update({c:cid})	
        cid +=1
    #print (aDictClassID)


   
	
    ###############################
    #   Make the model 
    ###############################
    aDict = config['stats']['class_labels']
	
	# classes to eval - correspond to the classes 
	#                    into which the images files are split
	#                    i.e., 'male', 'female', 'junk'
    classes_to_eval = list(aDict.keys())    
    # classes stat - to collected statistics of that's been 
	#                picked up/determined on an image
    classes_stat = classes_to_eval + ["unknown"]	
    run_stat         = np.zeros((len(classes_to_eval), len(classes_stat)))
    run_stat_percent = np.zeros((len(classes_to_eval), len(classes_stat)))

    aDictClassID = {}
    aDictFileCntByClass = {}   
    classes_to_eval = np.sort(classes_to_eval)
    print ("classes to eval: ", classes_to_eval) 
    cid = 0
    for c in classes_to_eval:
        aDictClassID.update({c:cid})	
        aDictFileCntByClass.update({c:0})    
        cid +=1


    aDict               = collections.OrderedDict(sorted(aDict.items()))
    aDictClassID        = collections.OrderedDict(sorted(aDictClassID.items()))
    aDictFileCntByClass = collections.OrderedDict(sorted(aDictFileCntByClass.items()))

    print ('aDict              : ', aDict)
    print ('aDictClassID       : ', aDictClassID)
    print ('aDictFileCntByClass: ', aDictFileCntByClass)
    
    

    yolo = YOLO(architecture        = config['model']['architecture'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################    

    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################

    pattern            = '*.jpg'
    log_file_trace     = 'log-' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.txt'
    log_file_err       = 'log-err-' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.txt'
    log_file_stat     = 'log-stat-' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.txt'

    str_log_trace      = ''
    str_log_err        = ''


        
    
    files = [f for f in os.listdir(image_path)  if os.path.isfile(os.path.join(image_path, f))]
    
    totalFiles = len(files)
    i =0 
        


    for name in files:
        
        if fnmatch(name, pattern):
            filename = os.path.join(image_path, name)
            
            str_log_trace += image_path + ',' + name
            #print ('fileName: ', filename)
            image = cv2.imread(filename)
            boxes = yolo.predict(image)


            # Filter only one box to show
            maxScore = 0
            for box in boxes:
                maxScore = max(maxScore, box.get_score())



            boxesTop = [box for box in boxes if box.get_score() == maxScore]
            
            
            for box in boxesTop:
                box.w  = min (.7, box.w)
                box.h  = min (.7, box.h) 
                print ('fileName: {} x={:.2f} y ={:.2f} w={:.2f} h={:.2f}'.format( filename,box.x,box.y, box.w, box.h))

            image = draw_boxes(image, boxesTop, config['model']['labels'])

            
            #sys.stdout.write(".")
            #sys.stdout.flush()

            

            tmp_s = np.zeros(len(classes_stat))
           
            str_finding = ''
            
            for box in boxesTop:
                box.w  = min (.7, box.w)
                box.h  = min (.7, box.h) 

                box_label = config['model']['labels'][box.get_label()]
                box_score = box.get_score()
                str_finding += ',' + box_label + ',' + str(box_score)
                clsID = -1
                for clsName, clsLabelLetter in aDict.items():
                    if (box_label[:1] == clsLabelLetter):
                        clsID = aDictClassID[clsName]                        
                        tmp_s[clsID] = max(box_score, tmp_s[clsID]) 
                        #break
            # if no classes been spotted in the image
            # mark the 'unknown' class as "discovered"
            if (np.sum(tmp_s) == 0):
                forcedBox = BoundBox(.5, .5, .5, .5, [0,0,0,0,0,0,0,.6], ['?'])
                image = draw_boxes(image, [forcedBox], ['uknown - nothing picked'])
                tmp_s[-1] =1 
            else:
                indexMax  = np.argmax(tmp_s)
                tmp_s = np.zeros(len(classes_stat))
                tmp_s[indexMax] = 1 

            filePrefix = '_'
            if (clsID ==0):
                filePrefix = 'f'
            elif (clsID ==1):
                filePrefix = 'j'
            elif (clsID ==2):
                filePrefix = 'm'

            clsID = np.argmax(tmp_s)
            print ("predictfld-mixed: class id: {}  {}".format(clsID, filename))          
                        
            str_log_trace += str_finding + '\n'
            
            cv2.imwrite(os.path.join(image_path +'/proc/'+ filePrefix +"/",  name), image)



    text_file = open( os.path.join(image_path +'/logs/', log_file_trace) , "w")
    text_file.write(str_log_trace)
    text_file.close()

    #text_file = open( os.path.join(image_path +'/logs/', log_file_err) , "w")
    #text_file = open(log_file_err, "w")
    #text_file.write(str_log_err)
    #text_file.close()
    

    #run_stat_percent
    

    print(' ')
    


     


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)

