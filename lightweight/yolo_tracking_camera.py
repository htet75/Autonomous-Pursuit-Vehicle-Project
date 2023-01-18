import cv2 as cv
import torch
import os
from models.experimental import attempt_load
from utils.datasets import letterbox, np
from utils.general import check_img_size, non_max_suppression, apply_classifier,scale_coords, xyxy2xywh
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier,TracedModel
from deep_sort import tracker
from PIL import Image
import random
import torch.backends.cudnn as cudnn
import imutils

show = True

tempImg = None
img = None
outputs = None
mot_tracker = tracker()
id = 0

classes = open("./names/coco.names".read().strip().split('\n'))
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
            
#==============EXIT CODES================
# 1: No Camera Input