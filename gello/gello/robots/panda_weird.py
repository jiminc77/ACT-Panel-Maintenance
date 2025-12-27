import time
from typing import Dict
from modules.config import Config

import numpy as np

from gello.robots.robot import Robot

from flask import Flask, request, jsonify
import numpy as np
import rospy
import time
import subprocess
from scipy.spatial.transform import Rotation as R
from absl import app, flags

from franka_msgs.msg import ErrorRecoveryActionGoal, FrankaState
from franka_gripper.msg import GraspActionGoal, MoveActionGoal
from serl_franka_controllers.msg import ZeroJacobian
from sensor_msgs.msg import JointState
import geometry_msgs.msg as geom_msg
from dynamic_reconfigure.client import Client as ReconfClient

import sys
sys.path.append('/home/demo-panda/catkin_ws/src')

print(sys.path)
from vc_controller.robot_infra.envs.franka_vc_env import FrankaVC

MAX_OPEN =0.09

class PandaRobot(Robot):
    """A class representing a UR robot."""

    def __init__(self, robot_ip: str = "100.97.47.74"):
       
        self.config_robot = '/home/demo-panda/catkin_ws/src/vc_controller/robot_infra/configs/robot_params.yaml'
        self.gripper_state = 'open'
        
        if self.gripper_state == 'close':
            self.start_gripper = 0
        else:
            self.start_gripper = 1
            
        self.config_robot = Config(self.config_robot).get_config()
        self.control = FrankaVC(config_robot=self.config_robot, hz=100, start_gripper=self.start_gripper)
        
        self.control.precision_mode()
        
        self.move_to_initial_pose()
        
        # self.control.change_to_joint_controller()
        
        
        
        breakpoint()

        # from polymetis import GripperInterface, RobotInterface

        # self.robot = RobotInterface(
        #     ip_address=robot_ip,
        # )
        # self.gripper = GripperInterface(
        #     ip_address="localhost",
        # )
        # self.robot.go_home()
        # self.robot.start_joint_impedance()
        # self.gripper.goto(width=MAX_OPEN, speed=255, force=255)
        # time.sleep(1)

    def move_to_initial_pose(self):
        self.init_pos = np.array([0.5787369523386803, 
                                  -0.0009511671965929064, 
                                  0.3728466930398651, 
                                  0.9992724297958465, 
                                  0.008269629160297989, 
                                  0.03314168450532647, 
                                  0.01696623209807298])
        self.control.move_to_pos(self.init_pos)
        time.sleep(2.0)
        
   
    def num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        return 8

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        # robot_joints = self.robot.get_joint_positions()
        robot_joints = self.control.get_joint_pos()['q']
        gripper_pos = self.control.gripper_dist.item()
        pos = np.append(robot_joints, gripper_pos.width / MAX_OPEN)
        return pos

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (np.ndarray): The state to command the leader robot to.
        """
        import torch

        # self.robot.update_desired_joint_positions(torch.tensor(joint_state[:-1]))
        # self.gripper.goto(width=(MAX_OPEN * (1 - joint_state[-1])), speed=1, force=1)
        # self.control.
        self.control.control_gripper(width=(MAX_OPEN * (1 - joint_state[-1])))

    def get_observations(self) -> Dict[str, np.ndarray]:
        joints = self.get_joint_state()
        pos_quat = np.zeros(7)
        gripper_pos = np.array([joints[-1]])
        return {
            "joint_positions": joints,
            "joint_velocities": joints,
            "ee_pos_quat": pos_quat,
            "gripper_position": gripper_pos,
        }


def main():
    robot = PandaRobot()
    current_joints = robot.get_joint_state()
    # move a small delta 0.1 rad
    move_joints = current_joints + 0.05
    # make last joint (gripper) closed
    move_joints[-1] = 0.5
    time.sleep(1)
    m = 0.09
    robot.gripper.goto(1 * m, speed=255, force=255)
    time.sleep(1)
    robot.gripper.goto(1.05 * m, speed=255, force=255)
    time.sleep(1)
    robot.gripper.goto(1.1 * m, speed=255, force=255)
    time.sleep(1)


if __name__ == "__main__":
    main()
