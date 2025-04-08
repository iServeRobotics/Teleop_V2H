import argparse
import h5py
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from inspire_hand import InspireHand
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
    X = int(pos[0])
    Y = int(pos[1])
    Z = int(pos[2])
    RX = int(euler_angles_degrees[0])
    RY = int(euler_angles_degrees[1])
    RZ = int(euler_angles_degrees[2])

    return X, Y, Z, RX, RY, RZ

def convert_pose_to_robot_command(pose):

    return pose[:3].tolist(), pose[3:6].tolist()


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


def load_synchronized_data(h5_path, csv_path):
    """
    Load and synchronize arm poses and finger angles using timestamps
    
    Returns:
        list: List of dicts with {
            'timestamp': float,
            'qpos': np.array (arm pose),
            'finger_angles': np.array
        }
    """
    # Load arm data from HDF5
    with h5py.File(h5_path, 'r') as f:
        arm_timestamps = f['observations/observations_ts'][:]
        teleop_pose = f['observations/teleop_pose'][:]
    
    # Load finger data from CSV (timestamp as first column)
    finger_data = pd.read_csv(csv_path, header=None)
    finger_timestamps = finger_data.iloc[:, 0].values
    finger_angles = finger_data.iloc[:, 1:-1].values
    
    # Find closest finger angles for each arm pose
    synchronized_data = []
    for i, (ts, tele_pose) in enumerate(zip(arm_timestamps, teleop_pose)):
        # Find index of closest timestamp in finger data
        idx = np.argmin(np.abs(finger_timestamps - ts))
        synchronized_data.append({
            'timestamp': ts,
            'teleop_pose': tele_pose,
            'finger_angles': finger_angles[idx]
        })
    
    print(f"Loaded {len(synchronized_data)} synchronized poses")
    return synchronized_data

def plot_trajectory(synchronized_data, filename=""):
    """3D plot of arm trajectory with finger angle visualization"""
    # Extract arm positions
    positions = np.array([d['teleop_pose'][:3] for d in synchronized_data])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 2)
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(gs[:, 0], projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', alpha=0.7)
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=100, label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=100, label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Arm Trajectory')
    ax1.legend()
    
    # Finger angles plot
    finger_angles = np.array([d['finger_angles'] for d in synchronized_data])
    print(f"finger angles dimension: {np.shape(finger_angles)}")
    print(finger_angles)
    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(finger_angles.shape[1]):
        ax2.plot(finger_angles[:, i], label=f'Finger {i+1}')
    ax2.set_title('Finger Angles Over Time')
    ax2.set_ylabel('Angle (deg)')
    ax2.legend()
    
    plt.suptitle(f'Trajectory Preview: {filename}')
    plt.tight_layout()
    plt.show()

def replay_synchronized(piper, synchronized_data, delay=0.5):
    """Replay synchronized arm and finger movements"""
    print(f"Replaying {len(synchronized_data)} synchronized poses...")
    
    for i, data in enumerate(synchronized_data):
        print(f"Pose {i+1}/{len(synchronized_data)} | Time: {data['timestamp']:.3f}s")
        
        try:
            # Convert and send arm command
            arm_cmd = piper.convert_qpos_to_command(data['qpos'])
            piper.send_arm_command(arm_cmd)
            
            # Send finger command
            piper.send_finger_command(data['finger_angles'])
            
            time.sleep(delay)
        except Exception as e:
            print(f"Error at pose {i+1}: {e}")
            break

def main():
    parser = argparse.ArgumentParser(description='Replay synchronized arm and finger trajectories')
    parser.add_argument('h5_file', help='Path to HDF5 file with arm poses')
    parser.add_argument('csv_file', help='Path to CSV file with finger angles')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay between poses (seconds)')
    parser.add_argument('--preview-only', action='store_true', help='Only preview without execution')
    args = parser.parse_args()

    # Load and synchronize data
    try:
        data = load_synchronized_data(args.h5_file, args.csv_file)
        if not data:
            print("No valid synchronized data found")
            return
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    if args.preview_only:
        print("Preview complete - execution skipped")
        # Visualize
        plot_trajectory(data, args.h5_file)
        return

    # Execute after confirmation
    response = input("Execute synchronized trajectory? (y/n): ").lower()
    if response != 'y':
        print("Execution cancelled")
        return

    # Execute on robot
    print(f"Starting the robot.")
    piper = C_PiperInterface("can0") # change this to the can port on V2H (likely can2)
    piper.ConnectPort()
    piper.EnableArm(7)
    enable_fun(piper=piper)

    dex_hand = InspireHand("/dev/ttyUSB0")
    

    poses = np.array([d['teleop_pose'] for d in data])
    finger_angles = np.array([d['finger_angles'] for d in data])

    print(poses)

    if np.shape(poses)[1] == 6:
        for pose, angles in zip(poses, finger_angles):
            print(f"pose: {pose}, finger angle: {angles}")
            pos, euler = convert_pose_to_robot_command(pose)
            # send command to robot
            X,Y,Z,RX,RY,RZ = get_pose_cmd(pos, euler)
            # print(f"x: {X}, y: {Y}, z: {Z}, rx: {RX}, ry: {RY}, rz: {RZ}")
            piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
            piper.EndPoseCtrl(X,Y,Z,RX,RY,RZ)
            dex_hand.gesture_by_angles(angles[0], angles[1], angles[2], angles[3], angles[4], angles[5])
            time.sleep(args.delay)
            
    print("Finished replay.")

if __name__ == "__main__":
    main()