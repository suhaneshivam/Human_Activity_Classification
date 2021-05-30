import numpy as np
import argparse
import imutils
import sys
import cv2
from collections import deque


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained human activity recognition model")
ap.add_argument("-c", "--classes", required=True,
	help="path to class labels file")
ap.add_argument("-i", "--input", type=str, default="",
	help="optional path to video file")
args = vars(ap.parse_args())

CLASSES = open(args["classes"] ,"r").read().strip().split("\n")
SAMPLE_DURATION = 16
SAMPLE_SIZE = 112


frames = deque(maxlen=SAMPLE_DURATION)

print("[INFO] loading human activity recognition model...")
net = cv2.dnn.readNet(args["model"])

#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)


vs =cv2.VideoCapture(args["input"] if args["input"] else 0)
#vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
i = 0

while True:

	(grabbed ,frame) = vs.read()

	if not grabbed:
		break
	frame = imutils.resize(frame ,width = 400)
	frames.append(frame)

	if len(frame)< SAMPLE_DURATION:
		continue

	#frame ,scaling factor,resizing size ,mean channel wise ,wheather want to swap RGB to BGR or not ,wheather first resize the image to min of resizing dims and then crop the image to adjust max dim.
	blob = cv2.dnn.blobFromImages(frames ,1.0 ,(SAMPLE_SIZE ,SAMPLE_SIZE), (114.7748, 107.7354, 99.4750) ,swapRB = 1 ,crop = True)
	#blob.shape :(16, 3, 112, 112) ,where 3 is the number of channels and 16 are no of frames from video and (112 ,112) are the height and  width of the frame.Here we are swaping  "channels" and "no. of frames" axes.And we are additionally adding a dimension axes in front,indicating the batch size(i.e. the no of video classification).Network we are using expects the input in this form only.
	blob = np.transpose(blob ,(1,0,2,3))
	blob = np.expand_dims(blob ,axis = 0)
	#blob.shape :(1 ,3, 16, 112, 112)
	#print(blob.shape)

	net.setInput(blob)
	outputs = net.forward()
	# print(outputs)
	# print(np.argmax(outputs))

	label = CLASSES[np.argmax(outputs)]

	cv2.rectangle(frame ,(0 ,0),(300 ,40) ,(0 ,0 ,0) ,-1)
	cv2.putText(frame ,str(label) ,(10 ,25) ,cv2.FONT_HERSHEY_SIMPLEX ,0.8 ,(255 ,255 ,255) ,2)
	cv2.imshow("Activity recognition" ,frame)


	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
