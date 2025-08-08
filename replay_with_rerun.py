import h5py
import rerun as rr
import time
import numpy as np

def inspect_h5_file(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        # Print the structure of the file
        def print_structure(name, obj):
            print(f"{name}: {obj}")
        
        h5_file.visititems(print_structure)


# Function to load the HDF5 file and replay data with visualization using rerun
def load_and_replay_with_rerun(file_path, num_reruns=1):
    # Open the HDF5 file
    with h5py.File(file_path, 'r') as h5_file:
        # Load data from the HDF5 file
        cam0_images = h5_file['observations/images/cam0'][:]  # Shape: (timesteps, height, width, channels)
        cam0_depth = h5_file['cam0_depth'][:]
        cam1_images = h5_file['observations/images/cam1'][:]
        cam1_depth = h5_file['cam1_depth'][:]
        # cam2_images = h5_file['observations/images/cam2'][:]
        actions = h5_file['action'][:]  # Shape: (timesteps, action_dims)
        qpos = h5_file['observations/qpos'][:]
        joints = qpos[:,7:13] / 1e3
        
        # Initialize Rerun logging
        rr.init("rerun_example_image", spawn=True)
        # rr.connect()

        # Replay the data for the specified number of reruns
        for run in range(num_reruns):
            print(f"Replay Run: {run + 1}/{num_reruns}")

            # Replay the data for each timestep
            for i in range(len(cam0_images)):
                rr.set_time_sequence("image0", i)
                # Add the image to Rerun (assuming image is a 2D array, if 3D adjust accordingly)
                rr.log("observations/image0", rr.Image(cam0_images[i]))  # Log the image as it is
                # rr.set_time_sequence("image1", i)
                rr.log("observations/image1", rr.Image(cam1_images[i]))
                # rr.log("observations/image2", rr.Image(cam2_images[i]))

                # augmented depth images:
                rr.log("cam0_depth", rr.Image(cam0_depth[i]))
                rr.log("cam1_depth", rr.Image(cam1_depth[i]))

                # Pause to simulate the replay
                time.sleep(0.1)  # Adjust this to control the speed of the replay

                # Get the action values (x, y, z) and log them in 3D space
                end_pose_xyz = np.array(actions[i, :3])  # Only use the first three columns (x, y, z)
                rr.log("robot/arm_endpose", rr.Points3D(end_pose_xyz))  # Log the point in 3D

                # Plot all actions (x, y, z, rx, ry, rz)
                rr.log("action/x", rr.Scalar(actions[i,0]))
                rr.log("action/y", rr.Scalar(actions[i,1]))
                rr.log("action/z", rr.Scalar(actions[i,2]))
                rr.log("action/rx", rr.Scalar(actions[i,3]))
                rr.log("action/ry", rr.Scalar(actions[i,4]))
                rr.log("action/rz", rr.Scalar(actions[i,5]))

                # Plot all joints
                rr.log("state/joint1", rr.Scalar(joints[i,0]))
                rr.log("state/joint2", rr.Scalar(joints[i,1]))
                rr.log("state/joint3", rr.Scalar(joints[i,2]))
                rr.log("state/joint4", rr.Scalar(joints[i,3]))
                rr.log("state/joint5", rr.Scalar(joints[i,4]))
                rr.log("state/joint6", rr.Scalar(joints[i,5]))
            print(f"Replay Run {run + 1} finished.")

        # Keep the rerun visualization window open
        # rr.wait_for_message()

if __name__ == "__main__":
    # Replace with your actual file path
    # h5_file_path = 'data_1741294139.433354_iserve.h5'
    # h5_file_path = 'episode_0.hdf5'
    h5_file_path = 'pick_toy/data_1741297683.8608696_iserve.h5'
    # h5_file_path = 'pick_toy/data_1741297616.6419966_iserve.h5'
    
    # Define how many times you want the replay to run
    num_reruns = 1  # Adjust this value as needed
    
    inspect_h5_file(h5_file_path)
    # Load the file and start the replay with visualization using rerun
    load_and_replay_with_rerun(h5_file_path, num_reruns)
