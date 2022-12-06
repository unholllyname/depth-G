import numpy as np
import time
import cv2
import os
import imutils
import subprocess
from gtts import gTTS 
from pydub import AudioSegment
import IPython.display as ipd
#from utils import visualization_utils as vis_util
AudioSegment.converter = "project"

# load the COCO class labels our YOLO model was trained on
LABELS = open("coco_copy.names").read().strip().split("\n")

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# initialize
cap = cv2.VideoCapture(0)
frame_count = 0
start = time.time()
first = True
frames = []

    
while True:
	frame_count += 1
    # Capture frame-by-frameq
	ret, frame = cap.read()
	frame = cv2.flip(frame,1)
	frames.append(frame)

	if frame_count == 300:
		break
        
	if ret:
		key = cv2.waitKey(1)
		if frame_count % 60 == 0:
			end = time.time()
			# grab the frame dimensions and convert it to a blob
			(H, W) = frame.shape[:2]
			# construct a blob from the input image and then perform a forward
			# pass of the YOLO object detector, giving us our bounding boxes and
			# associated probabilities
			blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
				swapRB=True, crop=False)
			net.setInput(blob)
			layerOutputs = net.forward(ln)

			# initialize our lists of detected bounding boxes, confidences, and
			# class IDs, respectively
			boxes = []
			confidences = []
			classIDs = []
			centers = []
			# loop over each of the layer outputs
			for output in layerOutputs:
				# loop over each of the detections
				for detection in output:
					# extract the class ID and confidence (i.e., probability) of
					# the current object detection
					scores = detection[5:]
					classID = np.argmax(scores)
					confidence = scores[classID]

					# filter out weak predictions by ensuring the detected
					# probability is greater than the minimum probability
					if confidence > 0.5:
						# scale the bounding box coordinates back relative to the
						# size of the image, keeping in mind that YOLO actually
						# returns the center (x, y)-coordinates of the bounding
						# box followed by the boxes' width and height
						box = detection[0:4] * np.array([W, H, W, H])
						(centerX, centerY, width, height) = box.astype("int")

						# use the center (x, y)-coordinates to derive the top and
						# and left corner of the bounding box
						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))

						# update our list of bounding box coordinates, confidences,
						# and class IDs
						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						classIDs.append(classID)
						centers.append((centerX, centerY))

			# apply non-maxima suppression to suppress weak, overlapping bounding
			# boxes
			idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
			#draw bounding boxes on the image
			colors= np.random.uniform(0,255, size=(len(LABELS),3))
			for i in idxs:
				i=i
				x,y,w,h = boxes[i]
				classID=classIDs[i]
				color=colors[classID]   
				cv2.rectangle(frame, (round(x),round(y)), (round(x+w), round(y+h)),color,2)
				label="%s: %.2f" %(LABELS[classID], confidences[i])
				cv2.putText(frame, label, (x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,1, color,2)
			cv2.imshow("Object detection",frame)
			cv2.waitKey(50)
			texts = []

			# ensure at least one detection exists
			if len(idxs) > 0:
				# loop over the indexes we are keeping
				for i in idxs.flatten():
					# find positions
					centerX, centerY = centers[i][0], centers[i][1]
					if centerX <= W/3:
						W_pos = "kiri "
					elif centerX <= (W/3 * 2):
						W_pos = "tengah "
					else:
						W_pos = "kanan "
					
					if centerY <= H/3:
						H_pos = "atas "
					elif centerY <= (H/3 * 2):
						H_pos = "tengah "
					else:
						H_pos = "bawah "
					texts.append(H_pos + W_pos + LABELS[classIDs[i]])
			print(texts)
			
			if texts:
				description = ', '.join(texts)
				tts = gTTS(description, lang='id')
				tts.save('tts.mp3')
				ipd.Audio('tts.mp3')
				#tts = AudioSegment.from_mp3("tts.mp3")
				subprocess.call(["ffplay", "-nodisp", "-autoexit", "tts.mp3"])
cap.release()
cv2.destroyAllWindows()
# os.remove("tts.mp3")