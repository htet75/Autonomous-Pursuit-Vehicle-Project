import os
import cv2 as cv
from deep_sort import tracker

import time
import numpy as np
import argparse
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", type=str, required=True,
	help="path to YOLO Weights file, ends in .weights; string")
ap.add_argument("-c", "--config", type=str, required=True,
	help="path to YOLO Config file, ends in .cfg; string")
ap.add_argument("-l", "--labels", type=str, required=True,
	help="path to Labels file, ends in .names; string")
ap.add_argument("-p", "--videoPath", type=str,
	help="path to video; ends in .mp4, can have multiple files; string")
ap.add_argument("-b", "--webcam", type=str, default="False",
    help="specify if you are using a camera. leave empty/False if using video")
ap.add_argument("-o", "--outputPath", type=str, default="./mot_vid",
	help="path to output folder; string; default=./out_vids")
ap.add_argument("-s", "--showImages", type=str, default="False",
    help="specify if you want to show the images as they are processing; True or False; default=False"
)
args = vars(ap.parse_args())
write = True
show = False

tempImg = None
img = None
outputs = None
mot_tracker = tracker()
id = 0

classes = open(args["labels"]).read().strip().split('\n')
np.random.seed()
colors = np.random.randint(0, 255, size = (len(classes),3), dtype='uint8')

net = cv.dnn.readNetFromDarknet(args["config"], args["weights"])
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

def load_image(frame_orig):
    frame = frame_orig.copy()
    
    blob = cv.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    outputs = net.forward(ln)

    outputs = np.vstack(outputs)

    post_process(frame, outputs, 0.5)
    return frame

def post_process(img, outputs, conf):
    global id
    H, W = img.shape[:2]

    boxes = []
    confidences = []
    classIDs = []
    motInput = []
    for output in outputs:
        scores = output[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > conf:
            if classID != 0:
                continue
            x, y, w, h = output[:4] * np.array([W, H, W, H])
            p0 = int(x - w//2), int(y - h//2)
            boxes.append([*p0, int(w), int(h)])
            motInput.append([*p0, int(w)+p0[0], int(h)+p0[1], confidence])
            confidences.append(float(confidence))
            classIDs.append(classID)
    indices = cv.dnn.NMSBoxes(boxes, confidences, conf, conf-0.2)
    motInput = np.array([motInput[x] for x in indices])
    track_bbs_ids = mot_tracker.update(motInput) 

    if len(track_bbs_ids) > 0:
        for i in range(len(track_bbs_ids)):
            (x, y) = (int(track_bbs_ids[i][0]), int(track_bbs_ids[i][1]))
            (w, h) = (int(track_bbs_ids[i][2]-x), int(track_bbs_ids[i][3]-y))
            color = [int(c) for c in colors[classIDs[i]]]
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}".format(track_bbs_ids[i][4])
            id = max(id, track_bbs_ids[i][4])
            cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv.putText(img, "People: " + str(int(id)), (400,50), cv.FONT_HERSHEY_SIMPLEX, 2, color, 2)
if args["webcam"] == "True":
    cap = cv.VideoCapture(0)
elif args["videoPath"]:
    cap = cv.VideoCapture(args["videoPath"][0])
else:
    print("No input was provided")
    exit(1)

fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter(args["outputPath"] + '/' + 'MOTS20-09-result.mp4', fourcc, 30.0, (960, 540))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Ya done messed up")
        break
    newFrame = load_image(frame)
    if args["showImages"] == "True":
        cv.imshow("frame", newFrame)
    if cv.waitKey(1) == ord('q'):
        break
    out.write(newFrame)

cap.release()
out.release()