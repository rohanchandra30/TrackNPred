# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os


# assuming yolo dir in same dir as this file
YOLO_DIR = os.path.join("resources/yolo-coco")
CONF = 0.5
NMS = .5

def detect(inputDir, framesDir, outputTxt):

	f = open(outputTxt, "w+")

	# derive the paths to the YOLO weights and model configuration
	weightsPath = os.path.sep.join([YOLO_DIR, "yolov3.weights"])
	configPath = os.path.sep.join([YOLO_DIR, "yolov3.cfg"])

	# load our YOLO object detector trained on COCO dataset (80 classes)
	# and determine only the *output* layer names that we need from YOLO
	print("[INFO] Loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# initialize the video stream, pointer to output video file, and
	# frame dimensions
	vs = cv2.VideoCapture(inputDir)
	(W, H) = (None, None)

	# try to determine the total number of frames in the video file
	try:
		prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
			else cv2.CAP_PROP_FRAME_COUNT
		total = int(vs.get(prop))
		print("[INFO] {} Total frames in video".format(total))

	# an error occurred while trying to determine the total
	# number of frames in the video file
	except:
		print("[INFO] Could not determine # of frames in video")
		print("[INFO] No approx. completion time can be provided")
		total = -1

	# loop over frames from the video file stream
	fid = 0
	while True:
		# read the next frame from the file
		(grabbed, frame) = vs.read()

		# save frames
		frame_savename = os.path.join(framesDir, "{0:06d}.jpg".format(fid).rjust(6, '0'))
		# print(frame_savename)
		cv2.imwrite(frame_savename, frame)

		# print(frame)

		# if the frame was not grabbed, then we have reached the end
		# of the stream
		if not grabbed:
			break

		# if the frame dimensions are empty, grab them
		if W is None or H is None:
			(H, W) = frame.shape[:2]

		# construct a blob from the input frame and then perform a forward
		# pass of the YOLO object detector, giving us our bounding boxes
		# and associated probabilities
		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
			swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()

		# initialize our lists of detected bounding boxes, confidences,
		# and class IDs, respectively
		boxes = []
		confidences = []
		classIDs = []

		# loop over each of the layer outputs
		for output in layerOutputs:
			# loop over each of the detections
			for detection in output:
				# extract the class ID and confidence (i.e., probability)
				# of the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				# filter out weak predictions by ensuring the detected
				# probability is greater than the minimum probability
				if confidence > CONF:
					# scale the bounding box coordinates back relative to
					# the size of the image, keeping in mind that YOLO
					# actually returns the center (x, y)-coordinates of
					# the bounding box followed by the boxes' width and
					# height
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")

					# use the center (x, y)-coordinates to derive the top
					# and and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					# update our list of bounding box coordinates,
					# confidences, and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

		# apply non-maxima suppression to suppress weak, overlapping
		# bounding boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONF,
			NMS)

		# ensure at least one detection exists
		if len(idxs) > 0:
			# loop over the indexes we are keeping
			for i in idxs.flatten():
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])

				f.write("{},-1,{},{},{},{},{},-1,-1,-1,{}\n".format(fid, x, y, w, h, confidences[i], i))


		if total > 0:
			elap = (end - start)
			print("[INFO] Single frame took {:.4f} seconds".format(elap))
			print("[INFO] Estimated total time to finish: {:.4f}".format(
				elap * total))

		fid += 1

	# release the file pointers
	print("[INFO] Cleaning up...")
	vs.release()
	f.close()
	print("[INFO] Dectection finished...")

