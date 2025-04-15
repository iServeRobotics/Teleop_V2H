import serial
import numpy as np
import time

# open: (170, 170, 170, 170, 30, 0)
# close: (150, 160, 170, 170, -10, 0)

class InspireHand:
	wait_for_serial = 0.001

	def __init__(self, port_name, timeout_seconds = 0.01):
		self.ser = serial.Serial(port_name, 115200, 8, serial.PARITY_NONE, serial.STOPBITS_ONE, timeout = timeout_seconds)
		# cmd: packet headers 0xeb, 0x90, hand_id (default 1), register_length+3, 0x12 (write instruction flag),
		#      Address L, Address H, Data, checksum
		self.clear_error()
		self.set_speed()
		self.set_force()


	def clear_error(self):
		cmd_clear_error = [0xeb, 0x90, 0x01, 0x04, 0x12, 0xec, 0x03, 0x01]
		cmd_clear_error.append(sum(cmd_clear_error[2:])%256)
		self.ser.write(serial.to_bytes(cmd_clear_error))
		return_data = self.ser.read(9)
		print("Dex Hand: clear error command sent.")

	def set_speed(self):
		cmd = [0xeb, 0x90, 0x01, 0x0f, 0x12, 0xda, 0x05, 0xe8, 0x03, 0xe8, 0x03, 0xe8, 0x03, 0xe8, 0x03, 0xe8, 0x03, 0xe8, 0x03, 0x83]
		self.ser.write(serial.to_bytes(cmd))
		print("Dex Hand: speed set to max.")
		return_data = self.ser.read(9)

	def set_force(self):
		cmd = [0xeb, 0x90, 0x01, 0x0f, 0x12, 0xf2, 0x05, 0xe8, 0x03, 0xe8, 0x03, 0xe8, 0x03, 0xe8, 0x03, 0xe8, 0x03, 0xe8, 0x03, 0x9b]
		self.ser.write(serial.to_bytes(cmd))
		print("Dex Hand: force set to max.")
		return_data = self.ser.read(9)

	def calibrate_force(self):
		cmd = [0xeb, 0x90, 0x01, 0x04, 0x12, 0xf1, 0x03, 0x01]
		cmd.append(sum(cmd[2:])%256)
		self.ser.write(serial.to_bytes(cmd))
		time.sleep(6)
		return_data = self.ser.read(10)

	def read_force(self):
		cmd = [0xeb, 0x90, 0x01, 0x04, 0x11, 0x2e, 0x06, 0x0c]
		cmd.append(sum(cmd[2:])%256)
		self.ser.write(serial.to_bytes(cmd))
		time.sleep(self.wait_for_serial)
		return_data = self.ser.read(20)
		# print(f"return data: {len(return_data)}")
		if len(return_data) < 20:
			print("Force read failed.")
			return [0, 0, 0, 0, 0, 0]
		little_finger_force = np.int16(return_data[8] << 8) + return_data[7]
		ring_finger_force = np.int16(return_data[10] << 8) + return_data[9]
		middle_finger_force = np.int16(return_data[12] << 8) + return_data[11]
		index_finger_force = np.int16(return_data[14] << 8) + return_data[13]
		thumb_bending_force = np.int16(return_data[16] << 8) + return_data[15]
		thumb_rotation_force = np.int16(return_data[18] << 8) + return_data[17]
		# print(f"force at little finger:  {little_finger_force}")
		# print(f"force at ring finger:  {ring_finger_force}")
		# print(f"force at middle finger:  {middle_finger_force}")
		# print(f"force at index finger:  {index_finger_force}")
		# print(f"force at thumb bending:  {thumb_bending_force}")
		# print(f"force at thumb rotation:  {thumb_rotation_force}")

		return np.array([little_finger_force, ring_finger_force, middle_finger_force, index_finger_force, thumb_bending_force, thumb_rotation_force])

	def gesture_by_angles(self, little_finger_angle, ring_finger_angle, middle_finger_angle, index_finger_angle, thumb_bending_angle, thumb_rotation_angle):
		# construct the serial command based on the finger angles
		cmd_angel_set = [235, 144, 1, 15, 18, 206, 5]
		# little_finger_angle = 170 # 19 to 176.7 degrees
		# ring_finger_angle = 150
		# middle_finger_angle = 150
		# index_finger_angle = 150
		# thumb_bending_angle = 12.6  # -13 to 53.6 degrees
		# thumb_rotation_angle = 165  # 90 to 165 degrees

		little_finger_value = int((little_finger_angle - 19) / 167.7 * 1000)
		little_finger_value = max(0, little_finger_value)
		little_finger_value = min(1000, little_finger_value)
		cmd_angel_set.append(np.int16(little_finger_value & 0xff))
		cmd_angel_set.append(np.int16(little_finger_value & 0xff00) >> 8)

		ring_finger_value = int((ring_finger_angle - 19) / 167.7 * 1000)
		ring_finger_value = max(0, ring_finger_value)
		ring_finger_value = min(1000, ring_finger_value)
		cmd_angel_set.append(np.int16(ring_finger_value & 0xff))
		cmd_angel_set.append(np.int16(ring_finger_value & 0xff00) >> 8)

		middle_finger_value = int((middle_finger_angle - 19) / 167.7 * 1000)
		middle_finger_value = max(0, middle_finger_value)
		middle_finger_value = min(1000, middle_finger_value)
		cmd_angel_set.append(np.int16(middle_finger_value & 0xff))
		cmd_angel_set.append(np.int16(middle_finger_value & 0xff00) >> 8)

		index_finger_value = int((index_finger_angle - 19) / 167.7 * 1000)
		index_finger_value = max(0, index_finger_value)
		index_finger_value = min(1000, index_finger_value)
		cmd_angel_set.append(np.int16(index_finger_value & 0xff))
		cmd_angel_set.append(np.int16(index_finger_value & 0xff00) >> 8)

		thumb_bending_value = int((thumb_bending_angle + 13) / 66.6 * 1000)
		thumb_bending_value = max(0, thumb_bending_value)
		thumb_bending_value = min(1000, thumb_bending_value)
		cmd_angel_set.append(np.int16(thumb_bending_value & 0xff))
		cmd_angel_set.append(np.int16(thumb_bending_value & 0xff00) >> 8)

		thumb_rotation_value = int((thumb_rotation_angle - 90) / 75 * 1000)
		thumb_rotation_value = max(0, thumb_rotation_value)
		thumb_rotation_value = min(1000, thumb_rotation_value)
		# thumb_rotation_value = thumb_rotation_angle
		cmd_angel_set.append(np.int16(thumb_rotation_value & 0xff))
		cmd_angel_set.append(np.int16(thumb_rotation_value & 0xff00) >> 8)


		checksum = sum(cmd_angel_set[2:]) & 0xff
		cmd_angel_set.append(checksum)
		print(cmd_angel_set)
		self.ser.write(cmd_angel_set)
		return_data = self.ser.read(9)
		# print("Dex Hand: gesture by angle command sent with following angles")
		# print(f"{little_finger_angle}, {ring_finger_angle}, {middle_finger_angle}, {index_finger_angle}, {thumb_bending_angle}, {thumb_rotation_angle}")


	def gesture_by_angles_force_controled(self, little_finger_angle, ring_finger_angle, middle_finger_angle, index_finger_angle, thumb_bending_angle, thumb_rotation_angle):
		self.gesture_by_angles(little_finger_angle, ring_finger_angle, middle_finger_angle, index_finger_angle, thumb_bending_angle, thumb_rotation_angle)
		initial_index_finger_angle = index_finger_angle
		force_threshold = 200
		while True:
			force_array = self.read_force()
			if force_array[3] > force_threshold:
				index_finger_angle += 0.25 * (force_array[3] / 200)
			else:
				index_finger_angle -= 5
				index_finger_angle = max(initial_index_finger_angle, index_finger_angle)
			self.gesture_by_angles(little_finger_angle, ring_finger_angle, middle_finger_angle, index_finger_angle, thumb_bending_angle, thumb_rotation_angle)
			time.sleep(0.001)
			# print("in the loop")

	def read_angles(self):
		cmd = [235, 144, 1, 4, 17, 206, 5, 12] #10, 6, 12]
		checksum = sum(cmd[2:]) & 0xff
		cmd.append(checksum)
		# print(cmd)
		self.ser.write(cmd)
		time.sleep(self.wait_for_serial)
		return_data = self.ser.read(20)
		print(f"return data: {len(return_data)}")

		little_finger_raw = np.int16(return_data[8] << 8) + return_data[7]
		ring_finger_raw = np.int16(return_data[10] << 8) + return_data[9]
		middle_finger_raw = np.int16(return_data[12] << 8) + return_data[11]
		index_finger_raw = np.int16(return_data[14] << 8) + return_data[13]
		thumb_bending_raw = np.int16(return_data[16] << 8) + return_data[15]
		thumb_rotation_raw = np.int16(return_data[18] << 8) + return_data[17]
		
		little_finger_angle = little_finger_raw / 1000.0 * 167.7 + 19
		ring_finger_angle = ring_finger_raw / 1000.0 * 167.7 + 19
		middle_finger_angle = middle_finger_raw / 1000.0 * 167.7 + 19
		index_finger_angle = index_finger_raw / 1000.0 * 167.7 + 19
		thumb_bending_angle = thumb_bending_raw / 1000.0 * 66.6 -13
		thumb_rotation_angle = thumb_rotation_raw / 1000.0 * 75 + 90

		return [little_finger_angle, ring_finger_angle, middle_finger_angle, index_finger_angle, thumb_bending_angle, thumb_rotation_angle]

	def open_hand(self):
		self.clear_error()
		time.sleep(self.wait_for_serial)
		self.gesture_by_angles(176.7, 176.7, 176.7, 176.7, 53.6, 90)
		time.sleep(self.wait_for_serial)
		return_data = self.ser.read(9)
		print(f"return data length: {len(return_data)}")

	def grab_bowl_1(self):
		self.gesture_by_angles(150, 160, 170, 170, -10, 90)
		time.sleep(self.wait_for_serial)
		return_data = self.ser.read(20)
		print(f"return data length: {len(return_data)}")

	def grab_bowl_2(self):
		self.gesture_by_angles(100, 110, 130, 140, -13, 90)
		time.sleep(self.wait_for_serial)
		return_data = self.ser.read(20)
		print(f"return data length: {len(return_data)}")

	def grab_bowl_3(self):
		self.gesture_by_angles(100, 110, 110, 110, -13, 90)
		time.sleep(self.wait_for_serial)
		return_data = self.ser.read(20)
		print(f"return data length: {len(return_data)}")

	def grab_bowl_4(self):
		self.gesture_by_angles(90, 110, 110, 110, 20, 90)
		time.sleep(self.wait_for_serial)
		return_data = self.ser.read(20)
		print(f"return data length: {len(return_data)}")

	def grab_plate_2(self):
		self.gesture_by_angles(120, 120, 120, 120, -12, 90)
		time.sleep(self.wait_for_serial)
		return_data = self.ser.read(20)
		print(f"return data length: {len(return_data)}")

	def grab_plate_1(self):
		self.gesture_by_angles(120, 120, 120, 120, 0, 90)
		time.sleep(self.wait_for_serial)
		return_data = self.ser.read(20)
		print(f"return data length: {len(return_data)}")

# grab bowl: 30, 30, 70, 150, -10, 1000)  open:  (30, 30, 70, 150, 50, 1000)

	def close_port(self):
		self.ser.close()