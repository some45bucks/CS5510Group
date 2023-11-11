# Problem 4, b & c
We followed the instructions found in this [repo](https://github.com/akchobby/ROS2_ORB_SLAM3) with some slight modifications during build to work with ORB_SLAM3. Our Build files were quite large so they are not included in this repo, but we can provide them if requested.
Our output files, and some screenshots showing the created maps for each problem are located in their respective sub-directory.
The ros_bags and some videos of the code running are located in the submitted google drive.
Since building all the dependencies and ORB_SLAM3 is somewhat difficult and depends on the hardware used, we will also submit some videos of our code working.

# Prerequisites
We installed the following prerequisites on a Ubuntu 22.04 VM:
 - OpenCV
 - ROS Humble

- Build ORB_SLAM3
  - Go to this [repo](https://github.com/akchobby/ORB_SLAM3) and follow build instruction, including the additional prerequisites listed there.
    -After cloning the repo, but before building, run this command 
```
sed -i 's/++11/++14/g' CMakeLists.txt
```

- Install related ROS2 package
```
sudo apt install ros-$ROS_DISTRO-vision-opencv && sudo apt install ros-$ROS_DISTRO-message-filters
```

# Building
The OrbSlam/colcon_ws sub-directory in our build files can be cloned to capture any further modifications.
To build, run this command:
```
colcon build --symlink-install --packages-select orbslam3
```

# How we Used
1. Source the workspace
```
source ~/colcon_ws/install/local_setup.bash
```
2. Open a terminal new terminal, start the camera node. 
We were using an Intel RealSense D435 Camera. We were having some hardware limitations, our laptops were not able to handle all of the ros2 topics, so this is the command we had to use to prevent the VM from crashing:
```
ros2 run realsense2_camera realsense2_camera_node --ros-args -p rgb_camera.profile:=640x480x6
```

3. In the terminal where the workspace was sourced, we ran this command to begin ORB_SLAM3, using only monocular vision:
```
ros2 run orbslam3 mono ~/OrbSlam/ORB_SLAM3/Vocabulary/ORBvoc.txt ~/OrbSlam/ORB_SLAM3/Examples/Monocular/RealSense_D435i.yaml --ros-args -r camera:=/color/image_raw
```
Due to our hardware limitations, we could only map the raw image data, which produced less than ideal results. Our mapping was extremely inaccurate due to this.

4. To record the data to a ros2 bag, we ran this command in a new terminal:
```
ros2 bag record color/image_raw
```

# Output
## Part b
Our output for this part is located in the partb sub-directory and google drive.
The CameraTrajectory.txt and KeyFrameTrajectory.txt files, and some screenshots are located in the partb sub-directory.
The ros2 bag files and some videos are located in the Q4 folder on google drive.

## Comments
Due to the hardware limitations we were experiencing, we were limited to only using ORB_SLAM3 to view and map data from raw image topics. This meant our mapping and localization was extremely inaccurate. ORB SLAM was easily able to find key points to map, but due to only seeing raw images, there was no way for ORB SLAM to track depth of those points, or to easily keep track of its position and orientation. This led to the keypoints on the map getting extremely scattered and likely un-navigatable. The room we mapped is ~200 sq ft, but you are unable to see where the objects and walls in the room begin or end.

## Part d
Our output for this part is located in the partd sub-directory and google drive.
The CameraTrajectory.txt and KeyFrameTrajectory.txt files, and some screenshots are located in the partd sub-directory.
The ros2 bag files are located in the Q4 folder on google drive.

## Comments
Whenever the camera lost sight of any key points, or was moved too fast, especially during rotation, ORB SLAM would create a new map and attempt to re-localize. Due to our limited hardware, this made it difficult to get a full 3 circuits in an outdoor space. Each time it would try to re-localize, the hardware would freeze for a few seconds, and would often times crash the VM and require a restart. Because of this, it was difficult to get a full 3 circuits in during 1 recording session. Our best attempt was nearly 1 full rotations, but we still got a good idea of how the mapping was affected by the outdoor space.

Due to the re-localization and re-mapping everytime we lost sight of key points, we had to choose an area with a lot of nearby objects. We chose an area with a few trees scattered around. The leaves on the trees and ground were easily picked up by ORB SLAM, but they were much more random than the objects that tend to be indoors, chairs, tables, etc. This resulted in a much more noisy map, essentially just points scattered all over the place with no real sense of mapping. This paired with the fact that we were only able to map from the raw image data and no field of depth, created an even more inaccurate and noisy map when compared to the indoor map. The indoor mapping was at least sort of able to find where objects were because they keypoints would remain comparatively clumped.

