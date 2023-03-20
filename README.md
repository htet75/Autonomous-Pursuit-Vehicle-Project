# EEC174-CUI

## Introduction

The goal of our project is to make an autonomous vehicle that detects and tracks one person from multiple people in real-time. The RC car will move closer to the
person with the trigger, in this case a stop sign, when they walk away and will rotate if they turn to the side. The race car will attempt to stay a certain 
distance away from the person with the trigger.

This repo implements the use of 2 repos:
- https://github.com/wang-xinyu/tensorrtx allows us to run yolov7 at a higher FPS in comparison to typical pytorch yolo.

- https://github.com/NVIDIA-AI-IOT/trt_pose allows us to run pose estimation on Jetson devices. 

This repo implements https://github.com/deshwalmahesh/yolov7-deepsort-tracking with the use of a Jetson racecar and Tx2. The rc will follow and stop according to 
the speed of the person using an intel realsense depth camera d435. This repo makes modifications to the code for use with a remoteless rc car and Yolov7.

## Setup

1. First we must setup yolov7 working with TensorRT. More informatin can be found in the README.md of the yolov7_trt folder.
```
// download https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
cp {tensorrtx}/yolov7/gen_wts.py {WongKinYiu}/yolov7
cd {WongKinYiu}/yolov7
python gen_wts.py
// a file 'yolov7.wts' will be generated.

cd {tensorrtx}/yolov7/
// update kNumClass in config.h if your model is trained on custom dataset
mkdir build
cd build
cp {WongKinYiu}/yolov7/yolov7.wts {tensorrtx}/yolov7/build
cmake ..
make

// install python-tensorrt, pycuda, etc.
pip install tensorrt pycuda
```

You can check if your TensorRT works by running 
```
python yolov7_trt.py
```

2. In addition, we also used trt_pose in order to implement pose estimation. trt_pose uses a couple of dependencies that we need to install
```
// If you are using a Jetson Device, it recommends using https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
// otherwise pip install pytorch torchvision 

// install torch2trt
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
sudo python3 setup.py install --plugins

// Other required packages
sudo pip3 install tqdm cython pycocotools
sudo apt-get install python3-matplotlib

git clone https://github.com/NVIDIA-AI-IOT/trt_pose
cd ../trt_pose
sudo python3 setup.py install
```

If you want to test the finished installing the trt_pose, you can test it using the jupyter notebook found in the given repo
```
cd tasks/human_pose
jupyter live_demo.ipynb
```

3. In order to use the Intel DeepSense d435 camera that we have, you must install the pyrealsense library. On the Jetson TX2, that requires building from source.
This can be found at the repo https://github.com/IntelRealSense/librealsense and would require using CMake.

4. Once you have all the depencies installed, you can run our main code.
```
cd yolov7_trt
python3 yolov7_trt_cam.py
```

## References

https://github.com/deshwalmahesh/yolov7-deepsort-tracking

https://github.com/WongKinYiu/yolov7

https://github.com/nwojke/deep_sort.git

https://github.com/abrahamhadaf/Jetson-Autonomous-Car

https://github.com/wang-xinyu/tensorrtx

https://github.com/NVIDIA-AI-IOT/trt_pose
