#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from task.msg import Det
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2
from cv_bridge import CvBridge

br=CvBridge()

def decode_predictions(scores, geometry):
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	for y in range(0, numRows):
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]
		for x in range(0, numCols):
			if scoresData[x] < 0.5:
				continue
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
	return (rects, confidences)

def callback(data):
    rospy.loginfo("videofeed received...")
    image=br.imgmsg_to_cv2(data)
    #cv2.imshow("vidfromtopic",image)
    #cv2.waitKey(1)
    orig=image.copy()
    (origH, origW) = image.shape[:2]
    (newW, newH) = (640,640)
    rW = origW / float(newW)
    rH = origH / float(newH)
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]
    layerNames = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]
    net = cv2.dnn.readNet("/home/rishabh/Downloads/opencv-text-recognition/frozen_east_text_detection.pb")
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    results = []
    for (startX, startY, endX, endY) in boxes:
    	startX = int(startX * rW)
    	startY = int(startY * rH)
    	endX = int(endX * rW)
    	endY = int(endY * rH)

    	dX = int((endX - startX) * 0.0)#padding=0.0
    	dY = int((endY - startY) * 0.0)

    	startX = max(0, startX - dX)
    	startY = max(0, startY - dY)
    	endX = min(origW, endX + (dX * 2))
    	endY = min(origH, endY + (dY * 2))

    	roi = orig[startY:endY, startX:endX]

    	config = ("-l rus --oem 1 --psm 7")
    	text = pytesseract.image_to_string(roi, config=config)

    	results.append(((startX, startY, endX, endY), text))

    results = sorted(results, key=lambda r:r[0][1])
    pub=rospy.Publisher('/detection_result', Det, queue_size=10)
    rate = rospy.Rate(10)
    
    for ((startX, startY, endX, endY), tex) in results:
    	#print("OCR TEXT")
    	#print("========")
    	#print(type(tex))
    	#print(tex)
    	tex=tex.encode("utf-8")
    	#print(type(tex))
    	msg=Det()
    	msg.text=tex
    	msg.centre.x=float(startX+endX)/2
    	msg.centre.y=float(startY+endY)/2
    	msg.centre.z=0.0
    	#tex = "".join([c if ord(c) < 128 else "" for c in tex]).strip()
    	#output = orig.copy()
    	#cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)
    	#cv2.putText(output, tex, (startX, startY - 20),	cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    	#cv2.imshow("Text Detection", output)
    	#cv2.waitKey(1)
    	pub.publish(msg)
    	rospy.loginfo("message published...")
    	rate.sleep()
    
    

if __name__ == '__main__':
    rospy.init_node('detection', anonymous=True)
    sub=rospy.Subscriber('/videofeed',Image,callback)
    rospy.spin()
    