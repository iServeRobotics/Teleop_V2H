from SiriusCeptionBin import AsyncCeptionController, SiriusCeptionLib
import asyncio
import serial_asyncio
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import h5py
import cv2

from typing import (
    Optional,
)
import time
import math
from piper_sdk import *

def enable_fun(piper:C_PiperInterface):
	'''
	使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
	'''
	enable_flag = False
	# 设置超时时间（秒）
	timeout = 5
	# 记录进入循环前的时间
	start_time = time.time()
	elapsed_time_flag = False
	while not (enable_flag):
		elapsed_time = time.time() - start_time
		print("--------------------")
		enable_flag = piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
			piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
			piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
			piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
			piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
			piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
		print("使能状态:",enable_flag)
		piper.EnableArm(7)
		piper.GripperCtrl(0,1000,0x01, 0)
		print("--------------------")
		# 检查是否超过超时时间
		if elapsed_time > timeout:
			print("超时....")
			elapsed_time_flag = True
			enable_flag = True
			break
		time.sleep(1)
		pass
	if(elapsed_time_flag):
		print("程序自动使能超时,退出程序")
		exit(0)

def get_pose_cmd(pos, euler_angles_degrees):
	offset = [-0.2, 0, 0.5]
	factor = 1000
	X = round((offset[0]+pos[0,0])*600*factor)
	Y = round((offset[1]+pos[0,1])*600*factor)
	Z = round((offset[2]+pos[0,2])*600*factor)
	RX = round(euler_angles_degrees[0]*factor)
	RY = round(euler_angles_degrees[1]*factor)
	RZ = round(euler_angles_degrees[2]*factor)

	X = max(100*factor, X)
	X = min(700*factor, X)
	Y = max(-500*factor, Y)
	Y = min(500*factor, Y)
	Z = max(150*factor, Z)
	Z = min(700*factor, Z)

	return X, Y, Z, RX, RY, RZ

async def task():
	piper = C_PiperInterface("can0")
	piper.ConnectPort()
	piper.EnableArm(7)
	enable_fun(piper=piper)
	factor = 1000
	position = [
				200.0, \
				0.0, \
				300.0, \
				0, \
				150.0, \
				0, \
				0]
	X = round(position[0]*factor)
	Y = round(position[1]*factor)
	Z = round(position[2]*factor)
	RX = round(position[3]*factor)
	RY = round(position[4]*factor)
	RZ = round(position[5]*factor)
	piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
	piper.EndPoseCtrl(X,Y,Z,RX,RY,RZ)
	time.sleep(2.0)

	MPUReader = AsyncCeptionController("/dev/ttyUSB0") # this is the port for IMU teleoperation
	print("connecting ...")
	await MPUReader.connect()
	print("connected.")
	input("Please hit ENTER for calibration... (put your hand down, and relax).")
	sirius = SiriusCeptionLib(MPUReader)
	print("calibrating ...")
	await sirius.calibration()
	print("calibrated.")

	input("Please hit ENTER for calibration step two... (point your hand to the front).")
	yaw_angle = []
	for i in range(100):
		pos, rot = await sirius.getEndPos()
		yaw_angle.append(math.atan(pos[0,1]/pos[0,0]))
	yaw_offset = math.degrees(np.mean(yaw_angle))
	if pos[0,0] > 0 and pos[0,1] > 0:
		yaw_offset = yaw_offset
	elif pos[0,0] < 0 and pos[0,1] > 0:
		yaw_offset = 180.0 + yaw_offset
	elif pos[0,0] < 0 and pos[0,1] < 0:
		yaw_offset = 180.0 + yaw_offset
	elif pos[0,0] > 0 and pos[0,1] < 0:
		yaw_offset = yaw_offset
	print(f"yaw offset angle: {yaw_offset} degrees, {pos[0,0]}, {pos[0,1]}")

	r_matrix = R.from_euler('z', yaw_offset, degrees=True).as_matrix()

	h_comp_matrix = np.vstack((np.hstack((np.array(r_matrix), np.zeros(3).reshape(3,1))), np.array([0, 0, 0, 1]).reshape(1,4)))
	print(h_comp_matrix)

	action_list = []    # teleop pose
	robot_state_list = []  # robot end pose
	img0_list = []
	img1_list = []
	img_time_stamp_list = []

	camera_names = {'cam0', 'cam1'}
	cam0 = cv2.VideoCapture(0)
	cam1 = cv2.VideoCapture(2)
	data_dict = {
		'/observations/qpos': [],
		'/observations/qvel': [],
		'/action': [],
	}
	for cam_name in camera_names:
		data_dict[f'/observations/images/{cam_name}'] = []

	base_sleep_period = 0.01 # 100Hz
	camera_tic_factor = 3 # 30 times base sleep period, ~ 30Hz
	counter = 0

	start_time = time.time()

	while True: 
		try: # Retrieve end position and posture 
			pos, rot = await sirius.getEndPos() # 获取位置和旋转矩阵 
			p = np.array([pos[0,0], pos[0,1], pos[0,2]]).reshape(3,1)
			t = -np.matmul(rot, p)
			h_matrix = np.vstack((np.hstack((rot, t)), np.array([0,0,0,1]).reshape(1,4)))
			h_new = np.matmul(h_matrix, h_comp_matrix)
			new_p = -np.matmul(h_new[0:3,0:3].T, h_new[0:3, 3])
			
			# compensate for the end effector heading
			r_end = R.from_euler('z', -yaw_offset, degrees=True).as_matrix()
			h_end_comp_matrix = np.vstack((np.hstack((np.array(r_end), np.zeros(3).reshape(3,1))), np.array([0, 0, 0, 1]).reshape(1,4)))
			h_end = np.matmul(h_end_comp_matrix, h_matrix)

			
			# euler_angles_degrees = R.from_matrix(h_end[0:3,0:3]).as_euler('xyz', degrees=True)
			# euler_angles_degrees[1] += 90
			# euler_angles_degrees[2] += yaw_offset
			euler_angles_extrinsic_degrees = R.from_matrix(h_end[0:3,0:3]).as_euler('xyz', degrees=True)

			print(f"xyz: {new_p.reshape(1,3)} - euler angle extrinsic: {euler_angles_extrinsic_degrees}")
			# euler_angles_extrinsic_degrees[0] = 0
			adjusted_end_pose_orientation_degrees = [-90, 0, -90] # dummy value
			adjusted_end_pose_orientation_degrees[0] = -euler_angles_extrinsic_degrees[1] - 90
			adjusted_end_pose_orientation_degrees[1] = euler_angles_extrinsic_degrees[0] + 90
			adjusted_end_pose_orientation_degrees[2] = euler_angles_extrinsic_degrees[2] - 90
			print(f"adjusted : {adjusted_end_pose_orientation_degrees}")

			X,Y,Z,RX,RY,RZ = get_pose_cmd(new_p.reshape(1,3), adjusted_end_pose_orientation_degrees)
			print(X,Y,Z,RX,RY,RZ)
			piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
			piper.EndPoseCtrl(X,Y,Z,RX,RY,RZ)
			end_pose = piper.GetArmEndPoseMsgs().end_pose
			# print(f"x: {end_pose.X_axis}, y: {end_pose.Y_axis}, z: {end_pose.Z_axis}, rx: {end_pose.RX_axis}, ry: {end_pose.RY_axis}, rz: {end_pose.RZ_axis}")
			obs_act_ts = time.time()
			tele_raw_data = np.concatenate((np.array(new_p).flatten(), np.array(euler_angles_degrees)))
			print(f"teleop raw data: {tele_raw_data}")
			action_list.append([obs_act_ts, X,Y,Z,RX,RY,RZ])
			robot_state_list.append([obs_act_ts, end_pose.X_axis, end_pose.Y_axis, end_pose.Z_axis, end_pose.RX_axis, end_pose.RY_axis, end_pose.RZ_axis])
			# action = [end_pose.X_axis, end_pose.Y_axis, end_pose.Z_axis, end_pose.RX_axis, end_pose.RY_axis, end_pose.RZ_axis]
			# data_dict['/observations/qpos'].append()
			# data_dict['/observations/qvel'].append()
			# data_dict['/action'].append(action)
			# for cam_name in camera_names:
				# data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

			time.sleep(base_sleep_period)
			counter += 1
			if counter == camera_tic_factor:
				# get an camera image
				ret0, frame0 = cam0.read()
				ret1, frame1 = cam1.read()
				if not ret0 or not ret1:
					print("no image...")

				# if ret0:
				# 	cv2.imshow('Camera 0', frame0)
				# if ret1:
				# 	cv2.imshow('Camera 1', frame1)

				# # Break the loop if 'q' is pressed
				# if cv2.waitKey(1) & 0xFF == ord('q'):
				# 	break
				# data_dict[f'/observations/images/cam0'].append(frame0)
				# data_dict[f'/observations/images/cam1'].append(frame1)
				img0_list.append(frame0)
				img1_list.append(frame1)

				img_time_stamp_list.append(time.time())
				counter = 0

		except KeyboardInterrupt:
			output_file_name = 'data_' + str(time.time()) + '_iserve.h5'
			print("Process interrupted by user.") 
			print("Data Collection Ended, saving to file...")
			with h5py.File(output_file_name, 'w') as hf:
				obs = hf.create_group('observations')
				image = obs.create_group('images')
				hf.create_dataset("actions",  data=np.array(action_list))
				image.create_dataset("cam0", data=np.array(img0_list))
				image.create_dataset("cam1", data=np.array(img1_list))
				obs.create_dataset("robot_state", data=np.array(robot_state_list))
				obs.create_dataset("observations_ts", data=np.array(img_time_stamp_list))
				# for name, array in data_dict.items():
				# 	hf[name][...] = array
				print(f"Total {time.time() - start_time} seconds of data saved.")

				# print(img_list)
			break 
		except Exception as e: 
			print(f"An error occurred: {e}") 
			break 


async def main(): 
	print("Starting task...") 
	await task() 


if __name__ == "__main__": 
	try: 
		asyncio.run(main()) 
	except KeyboardInterrupt: 
		print("Program terminated by user.")





