JOINT_RESET_TARGET = [-0.07, -0.1, 0.0, -2.5, -0.1, 2.5, -0.6]
rosparam set /target_joint_positions '[-0.07, -0.1, 0.0, -2.5, -0.1, 2.5, -0.6]'
roslaunch serl_franka_controllers joint.launch robot_ip:=172.27.190.2 load_gripper:=true
