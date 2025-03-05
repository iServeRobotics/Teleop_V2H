from SiriusCeptionBin import AsyncCeptionController, SiriusCeptionLib
import asyncio
import serial_asyncio
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

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
	Y = max(-600*factor, Y)
	Y = min(600*factor, Y)
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
				90.0, \
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

	while True: 
		try: # Retrieve end position and posture 
			pos, rot = await sirius.getEndPos() # 获取位置和旋转矩阵 
			p = np.array([pos[0,0], pos[0,1], pos[0,2]]).reshape(3,1)
			t = -np.matmul(rot, p)
			h_matrix = np.vstack((np.hstack((rot, t)), np.array([0,0,0,1]).reshape(1,4)))
			h_new = np.matmul(h_matrix, h_comp_matrix)
			new_p = -np.matmul(h_new[0:3,0:3].T, h_new[0:3, 3])
			
			# comp_euler_angles = R.from_matrix(h_new[0:3,0:3]).as_euler('xyz', degrees=True)
			# print(f"compensated euler angle: {comp_euler_angles}")
			# compensate for the end effector heading
			r_end = R.from_euler('z', -yaw_offset, degrees=True).as_matrix()
			h_end_comp_matrix = np.vstack((np.hstack((np.array(r_end), np.zeros(3).reshape(3,1))), np.array([0, 0, 0, 1]).reshape(1,4)))
			h_end = np.matmul(h_end_comp_matrix, h_matrix)
			
			euler_angles_extrinsic_degrees = R.from_matrix(h_end[0:3,0:3]).as_euler('xyz', degrees=True)

			euler_angles_extrinsic_degrees[0] += 90
			euler_angles_extrinsic_degrees[1] += 90
			print(f"xyz: {new_p.reshape(1,3)} - euler angle extrinsic: {euler_angles_extrinsic_degrees}")
			euler_angles_extrinsic_degrees[0] = 0
			# r_endpose = R.from_euler('XYZ', euler_angles_extrinsic_degrees, degrees=True)
			# euler_angles_extrinsic_degrees = r_endpose.as_euler('xyz', degrees=True)

			X,Y,Z,RX,RY,RZ = get_pose_cmd(new_p.reshape(1,3), euler_angles_extrinsic_degrees)

			piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
			piper.EndPoseCtrl(X,Y,Z,RX,RY,RZ)
			end_pose_msg = piper.GetArmEndPoseMsgs()
			# print(end_pose_msg)
			time.sleep(0.01)
			
		except KeyboardInterrupt: 
			print("Process interrupted by user.") 
			print("Simulation Ended") 
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





