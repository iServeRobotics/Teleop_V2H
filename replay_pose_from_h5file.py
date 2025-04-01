import h5py
import numpy as np
import time
from piper_sdk import *
import argparse

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_trajectory(qpos_data):
    """
    Visualize the end-effector trajectory in 3D space
    
    Args:
        qpos_data: List of qpos arrays from HDF5 file
    """
    # Extract all x,y,z positions
    positions = []
    for qpos in qpos_data:
        if len(qpos) >= 3:  # Ensure we have at least x,y,z
            positions.append(qpos[:3])
    
    if not positions:
        print("No valid position data to plot")
        return
    
    positions = np.array(positions)
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(x, y, z, 'b-', linewidth=2, label='Trajectory')
    ax.scatter(x, y, z, c='r', marker='o', s=20, label='Poses')
    
    # Plot start and end points
    ax.scatter(x[0], y[0], z[0], c='g', marker='s', s=100, label='Start')
    ax.scatter(x[-1], y[-1], z[-1], c='k', marker='*', s=200, label='End')
    
    # Add labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('End-Effector Trajectory Preview')
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()

def read_poses_from_h5(file_path):
    """
    Read qpos data from HDF5 file where poses are stored under 'observations/qpos'.
    Handles both single dataset and group formats.
    
    Args:
        file_path: Path to HDF5 file
        
    Returns:
        list: List of qpos arrays
    """
    qpos_data = []
    try:
        with h5py.File(file_path, 'r') as f:
            # Navigate to observations/qpos
            if 'observations' in f and 'teleop_pose' in f['observations']:
                qpos_dataset = f['observations/teleop_pose']
                
                # Case 1: qpos is a single dataset with multiple timesteps
                if isinstance(qpos_dataset, h5py.Dataset):
                    # Assuming shape is (timesteps, qpos_dim)
                    for i in range(qpos_dataset.shape[0]):
                        qpos_data.append(qpos_dataset[i])
                
                # Case 2: qpos is a group containing individual poses
                elif isinstance(qpos_dataset, h5py.Group):
                    for key in sorted(qpos_dataset.keys(), key=lambda x: int(x)):
                        qpos_data.append(qpos_dataset[key][:])
                
                print(f"Loaded {len(qpos_data)} qpos samples from {file_path}")
            else:
                print("Could not find observations/qpos in HDF5 file")
                
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")
    
    return qpos_data

def replay_poses_on_robot(poses, delay_between_poses=1.0):
    """
    Replay a sequence of poses on the robot arm.
    
    Args:
        poses (list): List of poses to replay
        delay_between_poses (float): Time delay between poses in seconds
    """
    if not poses:
        print("No poses to replay")
        return
    
    print(f"Starting to replay {len(poses)} poses...")
    
    for i, pose in enumerate(poses):
        print(f"Moving to pose {i+1}/{len(poses)}")
        
        # Convert pose to robot command format if needed
        robot_command = convert_pose_to_robot_command(pose)
        
        # Send command to robot (implementation depends on robot API)
        send_pose_to_robot(robot_command)
        
        # Wait before moving to next pose
        time.sleep(delay_between_poses)
    
    print("Pose replay completed")


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
    X = int(pos[0])
    Y = int(pos[1])
    Z = int(pos[2])
    RX = int(euler_angles_degrees[0])
    RY = int(euler_angles_degrees[1])
    RZ = int(euler_angles_degrees[2])

    return X, Y, Z, RX, RY, RZ



# TODO: Implement these functions based on your specific robot API
def convert_pose_to_robot_command(pose):

    return pose[:3].tolist(), pose[3:6].tolist()


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Replay robot trajectories from HDF5 files')
    parser.add_argument('filename', type=str, help='Path to the HDF5 trajectory file')
    parser.add_argument('--delay', type=float, default=0.1, 
                       help='Delay between poses (seconds)')
    parser.add_argument('--preview-only', action='store_true',
                       help='Only preview trajectory without executing')
    args = parser.parse_args()


    # Load trajectory data
    try:
        poses = read_poses_from_h5(args.filename)
        if not poses:
            print(f"No valid trajectory data found in {args.filename}")
            return
    except Exception as e:
        print(f"Error loading file {args.filename}: {e}")
        return
    
    if args.preview_only:
        print("Preview complete - execution skipped")
        plot_trajectory(poses)
        return

    # Execute on robot after confirmation
    response = input("Execute trajectory on real robot? (y/n): ").lower()
    if response != 'y':
        print("Execution cancelled")
        return

    print(f"Starting the robot.")
    piper = C_PiperInterface("can0") # change this to the can port on V2H (likely can2)
    piper.ConnectPort()
    piper.EnableArm(7)
    enable_fun(piper=piper)
    
    if poses:
        for pose in poses:
            pos, euler = convert_pose_to_robot_command(pose)
            # send command to robot
            X,Y,Z,RX,RY,RZ = get_pose_cmd(pos, euler)
            print(f"x: {X}, y: {Y}, z: {Z}, rx: {RX}, ry: {RY}, rz: {RZ}")
            piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
            piper.EndPoseCtrl(X,Y,Z,RX,RY,RZ)
            time.sleep(args.delay)
            
    print("Finished replay.")

if __name__ == "__main__":
    main()