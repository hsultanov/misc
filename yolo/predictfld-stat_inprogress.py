#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
#from tqdm import tqdm
#from preprocessing import parse_annotation
#from utils import draw_boxes
#from frontend import YOLO
import json
from fnmatch import fnmatch
import datetime
import collections

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
    print (" - - - - - - - - - - - - - -" )
    aDict = config['stats']['class_labels']
    for k,v in aDict.items():
        print ("k, v: ", k, v)
    print (" - - - - - - - - - - - - - -" )
	
    classes_to_eval = list(aDict.keys())   
    
    classes_stat = classes_to_eval + ["unknown"]
    #classes_stat.append("unknown")
	
    print ("classes eval:", classes_to_eval)
    print ("classes stat:", classes_stat)  

    run_stat = np.zeros((len(classes_to_eval), len(classes_stat)))
    print ("run stat: ", run_stat)	
	
    aDictClassID = {}
    cid = 0
    for c in classes_to_eval:
        aDictClassID.update({c:cid})	
        cid +=1
    print (aDictClassID)



    print (" - - - - - - - - - - - - - - -")
    aDict = collections.OrderedDict(sorted(aDict.items()))
    aDictClassID = collections.OrderedDict(sorted(aDictClassID.items()))
    

    stat_str  = '+ '.ljust(10)    
    for cl in list(aDict.keys()):
        stat_str += cl.ljust(10)
    stat_str += 'unknown'.ljust(10) +  '\n'    

   # print ('      : ', stat_str)



    


   

    print (" - - - - -  Good Formatting  - - - - - - - - - -")

    for clsName, clsLabelLetter in aDict.items():        
        #
        # get id of the class
        #
        clsID_File = aDictClassID[clsName]        
        strToShow = '{0:<10}'.format(clsName) 
        for el in run_stat[clsID_File,:]:
            strToShow += '{0:<10}'.format(el)       
        #print (strToShow)        
        stat_str +=  strToShow + '\n'


    print (" - - - - -  FINAL  - - - - - - - - - -")
    print ( stat_str)




if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
