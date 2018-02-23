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
from flask import Flask, jsonify, request
from PIL import Image

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]=""

UPLOAD_FOLDER = os.path.basename('data')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

config_path  = 'conf.json'
weights_path = 'mos.h5'

with open(config_path) as config_buffer:    
	config = json.load(config_buffer)

#   Make the model 
yolo = YOLO(architecture        = config['model']['architecture'],
			input_size          = config['model']['input_size'], 
			labels              = config['model']['labels'], 
			max_box_per_image   = config['model']['max_box_per_image'],
			anchors             = config['model']['anchors'])

#   Load trained weights
print (weights_path)
yolo.load_weights(weights_path)

@app.route('/predict',methods=['POST'])
def predict():

	file = request.files['fileupload']
	filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
	file.save(filename)

	image_path = filename

	###############################
	#   Predict bounding boxes 
	###############################

	image = cv2.imread(image_path)
	boxes = yolo.predict(image)
	image = draw_boxes(image, boxes, config['model']['labels'])

	print (len(boxes), 'boxes are found')
	log_str = str(len(boxes)) + ' boxes are found\n'

	#cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)
	for box in boxes:
		box_label = config['model']['labels'][box.get_label()]
		box_score = box.get_score()
		log_str   += box_label + ',' + str(box_score) + '\n'
	
	os.remove(filename)

	return log_str

if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 5000))
	#run the app locally on the givn port
	app.run(host='0.0.0.0', port=port)
	#optional if we want to run in debugging mode
	#app.run(debug=True)