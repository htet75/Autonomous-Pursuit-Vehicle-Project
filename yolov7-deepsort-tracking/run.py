from detection_helpers import *
from tracking_helpers import *
from  bridge_wrapper import *
from PIL import Image

detector = Detector(classes = [11]) # it'll detect ONLY [person,horses,sports ball]. class = None means detect all classes. List info at: "data/coco.yaml"
detector.load_model('./weights/yolov7-tiny.pt',) # pass the path to the trained weight file

# Initialise  class that binds detector and tracker in one class
tracker = YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector)

# output = None will not save the output video
tracker.track_video("input.mp4", output="sam.avi", show_live = True, skip_frames = 0, count_objects = True, verbose=1)
