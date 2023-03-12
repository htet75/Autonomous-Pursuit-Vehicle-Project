import json
import trt_pose.coco
import trt_pose.models
import torch
from torch2trt import TRTModule
import time
import cv2
import torchvision.transforms as transforms
import PIL.Image


from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
from realsense_depth import *

def get_keypoints(image, human_pose, topology, object_counts, objects, normalized_peaks):
    """Get the keypoints from torch data and put into a dictionary where keys are keypoints
    and values the x,y coordinates. The coordinates will be interpreted on the image given.

    Args:
        image: cv2 image
        human_pose: json formatted file about the keypoints

    Returns:
        dictionary: dictionary where keys are keypoints and values are the x,y coordinates
    """
    height = image.shape[0]
    width = image.shape[1]
    keypoints = {}
    K = topology.shape[0]
    count = int(object_counts[0])

    for i in range(count):
        obj = objects[0][i]
        C = obj.shape[0]
        for j in range(C):
            k = int(obj[j])
            if k >= 0:
                peak = normalized_peaks[0][j][k]
                x = round(float(peak[1]) * width)
                y = round(float(peak[0]) * height)
                keypoints[human_pose["keypoints"][j]] = (x, y)

    return keypoints

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)



# print("Loading model")
# model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()

# MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'

# model.load_state_dict(torch.load(MODEL_WEIGHTS))

print("Loading optimized model")
# Load optimized model
OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))


# print("Running program")
# t0 = time.time()
# torch.cuda.current_stream().synchronize()
# for i in range(50):
#     y = model_trt(data)
# torch.cuda.current_stream().synchronize()
# t1 = time.time()

# print(50.0 / (t1 - t0))



num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

topology = trt_pose.coco.coco_category_to_topology(human_pose)
parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


def execute(image):
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    draw_objects(image, counts, objects, peaks)
    keypoints = get_keypoints(image, human_pose, topology, counts, objects, peaks)
    # print(keypoints)

    # if 'left_shoulder' in keypoints:
    #     print(type(keypoints['left_shoulder']))

    if (keypoints.get('left_shoulder', (0, 0))[1] > keypoints.get('left_wrist', (0, 224))[1] or keypoints.get('right_shoulder', (224, 0))[1] > keypoints.get('right_wrist', (224, 224))[1]):
        print("FOUND FOUND FOUND")
    else:
        print("NOT NOT NOT")
    # image_w = bgr8_to_jpeg(image[:, ::-1, :])
    return image

# Tester for pose estimation alone
if __name__ == "__main__":
    camera = DepthCamera()
    start = time.time()
    frame_num = 0
    while(True):

        ret, depth_frame, frame = camera.get_frame()
        frame_resize = cv2.resize(frame, (224, 224))
        # img = cv2.resize(frame[:,240:1680], (640, 480))
        
        result = execute(frame_resize)
        
        result = cv2.resize(result, (600, 480))
        end = time.time()
        frame_num+=1
        print("FPS is {:.2f}".format(frame_num/(end-start)))
        cv2.imshow("Recognition result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    

# camera.observe(execute, names='value')

# camera.unobserve_all()