# Data Collection through Teleop V2H

This repository provides a couple of Python scripts to facilitate data collection with lightweight coolabrate robot ARM as well as dexterous robot hand (optional), for robot learning purposes. All data is saved to HDF5 format, for efficiency and easy to integrate with training and deployment pipeline.

## Features

- Teleoperate piper ARM:
- Record Data from cameras and Robot ARM state.
- Replay a recorded data (WIP)

## Installation

Before running the script, you'll need to install the required Python packages.

### Step 1: Install conda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
```
Install by running the sh file. 

### Step 2: Create conda env and install python dependencies

```bash
conda create -n py311 python=3.11
conda activate py311
pip install numpy scipy h5py opencv-python-headless asyncio￼pyserial-asyncio piper_sdk pyserial
```

### Step 3: Run the telelop script to verify the teleoperation pipeline

Make sure can is setup correctly following tutorial on [https://github.com/agilexrobotics/piper_sdk]
Then run the following command:

```bash
python sample_teleop_piper.py
```
Then follow the instructions on the screen about the calibration (two steps) of the IMU based teleoperation devices. Once the calibration is finished please move your arm slowly and the robot arm should follow your motion. Once you are comfortable with the teleoperation, move forward to the data collection.


### Step 4: Data Collection

Make sure you have 2 USB cameras connected.

```bash
ls /dev/video*
```

If you saw video0 video1 video2 video3, video4, video5, then it means all three cameras are connected (video0 for hand control, video2 and video4 for data collection). If not please double check the cameras before proceed.

There is an known issue with the USB port, so adding additional verification as following (in python commandline):

```bash
import cv2

cam0 = cv2.VideoCapture(2)
cam1 = cv2.VideoCapture(4)

while True:
    ret0, img0 = cam0.read()
    ret1, img1 = cam1.read()

    if not ret0:
    	print("cam 0 is not available")
    else:
        print(img0)
    if not ret1:
    	print("cam 1 is not available")
    else:
    	print(img1)


```
Now if the script above shows no error, proceed to the following:

```bash
python data_collection_h5.py
```
And go through the same calibration steps in teleoperation, the data recording is automatically started after the calibration. Currently the data collection is set to record 500 observations (aobut a minute of data, 900+ MB) and will save a file with data_ as prefix and the current time (ROS time typed) with .hdf5 extension to the current path. Feel free to run the script as many times as you want, just be aware of the storage limitations.


### this is only a demo prototype and lots of stuff is WIP.