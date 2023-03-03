# EEC174-CUI

## Introduction
The goal of our project is to make an autonomous vehicle that detects and tracks people in real-time. The RC car will move closer to the person with the trigger, in
this case a stop sign, when they walk away and will rotate if they turn to the side. The race car will attempt to stay a certain distance away from the person with 
the trigger.

This repo implements https://github.com/deshwalmahesh/yolov7-deepsort-tracking with the use of a Jetson racecar and Tx2. The rc will follow and stop according to 
the speed of the person using an intel realsense depth camera d435. This repo makes modifications to the code for use with a remoteless rc and Yolov7.

##Setup
```
wget sdkafjlskdjfldskjf
pip3 install -r requirements.txt
``` 


##References
https://github.com/deshwalmahesh/yolov7-deepsort-tracking
https://github.com/WongKinYiu/yolov7
https://github.com/abrahamhadaf/Jetson-Autonomous-Car
