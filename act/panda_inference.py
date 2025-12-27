"""
Setup:
    1. Control PC (172.27.190.125): Run robot server (robot only)
       python experiments/launch_nodes.py --robot bimanual_panda --hostname 0.0.0.0 --robot_port 6001

    2. Learning PC: Run inference with webcams
       python panda_inference.py --mode voltage_check
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import tyro
import termcolor
import pickle
import traceback
import cv2

from gello.env import RobotEnv
from gello.zmq_core.robot_node import ZMQClientRobot
from gello.zmq_core.camera_node import ZMQClientCamera

from policy import ACTPolicy


def print_color(*args, color=None, attrs=(), **kwargs):
    """Print colored text to terminal"""
    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


@dataclass
class PandaInferenceArgs:
    """Arguments for Panda inference"""

    # Task configuration
    mode: str = "switch_off"  # 'voltage_check' or 'switch_off'
    ckpt_dir: Optional[str] = None  # Checkpoint directory (auto-selected if None)
    ckpt_name: str = "policy_best.ckpt"  # Checkpoint filename

    # Robot connection
    robot_port: int = 6001  # ZMQ port for robot server
    camera_port: Tuple[int, ...] = (7001, 8001)  # Camera ports for wrist1 and wrist2
    hostname: str = "172.27.190.125"  # IP of Panda control PC
    hz: int = 50  # Control frequency

    webcam_width: int = 640  # Webcam image width
    webcam_height: int = 480  # Webcam image height

    # Inference settings
    max_timesteps: int = 1000  # Maximum timesteps to run
    query_frequency: int = 1  # Run inference every N steps
    temporal_agg: bool = False  # Enable temporal aggregation

    # Debug and logging
    debug: bool = False
    log_frequency: int = 10  # Log progress every N steps
    
    # Visualization
    visualize: bool = True  # Enable real-time image visualization
    vis_frequency: int = 1  # Update visualization every N steps


class PandaInferenceEngine:
    """Inference engine for Panda robot using ACT policy"""

    def __init__(self, args: PandaInferenceArgs):
        self.args = args
        # Robot Hardware Indices: 0-6 (Left Arm), 8-14 (Right Arm)
        # Indices 7 and 15 are grippers (handled separately)
        self.arm_joint_indices = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14], dtype=int)
        self.action_dim = len(self.arm_joint_indices)  # 14

        # Auto-select checkpoint directory if not specified
        if args.ckpt_dir is None:
            args.ckpt_dir = f"./ckpts/{args.mode}"
            print_color(f"Auto-selected checkpoint directory: {args.ckpt_dir}", color="cyan")

        self.ckpt_dir = Path(args.ckpt_dir)
        self.ckpt_path = self.ckpt_dir / args.ckpt_name

        # Create output directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("inference_outputs") / f"ETRI_{args.mode}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print_color("="*70, color="cyan", attrs=("bold",))
        print_color("Panda Robot Inference (ACT Policy)", color="cyan", attrs=("bold",))
        print_color("="*70, color="cyan", attrs=("bold",))

        # Load checkpoint and stats
        self._load_checkpoint()

        # Initialize robot connection
        self._init_robot()

        # Initialize action buffer
        self.action_buffer = None  # Will store action chunks
        self.buffer_start_t = 0

        # Initialize temporal aggregation buffer and weights
        self.all_time_actions = None
        self.exp_weights = None  # [Optimization] Pre-computed weights
        
        self.black_image = None
        
        # Initialize visualization
        if self.args.visualize:
            print_color("  ✓ Real-time visualization enabled", color="green")

        print_color("="*70, color="green", attrs=("bold",))
        print_color("✓ Initialization Complete", color="green", attrs=("bold",))
        print_color("="*70, color="green", attrs=("bold",))

    def _load_checkpoint(self):
        """Load ACT policy and dataset statistics"""
        print_color("\n[1/3] Loading checkpoint...", color="cyan")

        # Check checkpoint exists
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_path}")

        # Load config.pkl
        config_path = self.ckpt_dir / "config.pkl"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, 'rb') as f:
            self.policy_config = pickle.load(f)

        # Load dataset_stats.pkl
        stats_path = self.ckpt_dir / "dataset_stats.pkl"
        if not stats_path.exists():
            raise FileNotFoundError(f"Stats not found: {stats_path}")

        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)

        # Extract key parameters
        configured_camera_names = self.policy_config.get('camera_names', ['wrist1'])
        if self.args.mode == "switch_off":
            self.camera_names = ['wrist1']
            self.black_image = np.zeros((3, self.args.webcam_height, self.args.webcam_width))
            print_color(" Switch_off mode: (wrist1 + black image)", color="yellow")
        else:
            self.camera_names = configured_camera_names
            self.black_image = None
        self.chunk_size = self.policy_config['policy_config']['num_queries']

        print_color(f"  ✓ Config loaded: {config_path}", color="green")
        print_color(f"  ✓ Stats loaded: {stats_path}", color="green")
        print_color(f"  ✓ Camera names (in hdf5): {self.camera_names}", color="green")
        print_color(f"  ✓ Chunk size: {self.chunk_size}", color="green")

        # Create policy
        policy_config = self.policy_config['policy_config']
        self.policy = ACTPolicy(policy_config)
        
        # Load weights
        checkpoint = torch.load(self.ckpt_path)
        loading_status = self.policy.deserialize(checkpoint)

        self.policy.cuda()
        self.policy.eval()

        # [Optimization] Apply torch.compile if available (PyTorch 2.0+)
        if hasattr(torch, "compile"):
            try:
                print_color("  ✓ Applying torch.compile() for optimization...", color="yellow")
                self.policy = torch.compile(self.policy)
            except Exception as e:
                print_color(f"  ⚠ torch.compile failed, using eager mode: {e}", color="red")

        print_color(f"  ✓ Policy loaded from: {self.ckpt_path}", color="green")
        print_color(f"  ✓ Loading status: {loading_status}", color="green")

        # Extract normalization parameters
        self.qpos_mean = self.stats['qpos_mean']
        self.qpos_std = self.stats['qpos_std']
        # self.action_mean = self.stats['action_mean'][self.arm_joint_indices]
        # self.action_std = self.stats['action_std'][self.arm_joint_indices]
        self.action_mean = self.stats['action_mean'][:14]
        self.action_std = self.stats['action_std'][:14]

        print_color(f"  ✓ Normalization stats ready", color="green")

    def visualize_images(self, obs: dict, step: int):
        """Visualize camera images in real-time using OpenCV"""
        if not self.args.visualize:
            return
        
        images_to_show = []
        titles = []
        
        for cam_name in self.camera_names:
            img = obs.get(cam_name)
            if img is not None and isinstance(img, np.ndarray) and img.size > 0:
                # Ensure image is in (H, W, 3) format with BGR for OpenCV
                if len(img.shape) == 3:
                    if img.shape[2] == 3:
                        # RGB to BGR for OpenCV
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        images_to_show.append(img_bgr)
                        titles.append(cam_name)
        
        if len(images_to_show) == 0:
            return
        
        # Create a combined image if multiple cameras
        if len(images_to_show) == 1:
            display_img = images_to_show[0]
        else:
            # Stack images horizontally
            display_img = np.hstack(images_to_show)
        
        # Add text overlay with step number
        cv2.putText(display_img, f"Step: {step}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add camera names if multiple cameras
        if len(images_to_show) > 1:
            img_width = images_to_show[0].shape[1]
            for i, title in enumerate(titles):
                x_pos = i * img_width + 10
                cv2.putText(display_img, title, (x_pos, display_img.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the image
        cv2.imshow("Panda Inference - Camera Feed", display_img)
        cv2.waitKey(1)  # Non-blocking wait

    def _init_robot(self):
        """Initialize ZMQ connection to Panda robot and webcams"""
        print_color(f"\n[2/3] Connecting to Panda robot at {self.args.hostname}:{self.args.robot_port}...", color="cyan")

        # Create ZMQ robot client
        self.robot_client = ZMQClientRobot(
            port=self.args.robot_port,
            host=self.args.hostname
        )

        # Create robot environment
        camera_dict = {}
        camera_port_map = {}
        if len(self.args.camera_port) > 0:
            camera_port_map['wrist2'] = self.args.camera_port[0]
        if len(self.args.camera_port) > 1:
            camera_port_map['wrist1'] = self.args.camera_port[1]
        for cam_name in self.camera_names:
            port = camera_port_map.get(cam_name)
            if port is None:
                raise ValueError(f"There are no port mapping '{cam_name}'. Please check args.camera_port")
            camera_dict[cam_name] = ZMQClientCamera(port=port, host=self.args.hostname)
        
        self.env = RobotEnv(
            robot=self.robot_client,
            control_rate_hz=self.args.hz,
            camera_dict=camera_dict,
            camera_read_interval=1
        )

        print_color(f"  ✓ Connected to robot server", color="green")
        # Get robot info
        num_dof = self.robot_client.num_dofs()
        print_color(f"  ✓ Robot DOF: {num_dof}", color="green")
        
        expected_dof = 16
        if num_dof != expected_dof:
            print_color(f"  ⚠ Warning: Expected {expected_dof} DOF, got {num_dof}", color="yellow")

        # Get initial observation
        obs = self.env.get_obs()
        print_color(f"  ✓ Observation keys: {list(obs.keys())}", color="green")

    def remap_qpos_to_training_order(self, qpos: np.ndarray) -> np.ndarray:
        """Remap robot qpos to match training data order (0-6, 8-14)."""
        return qpos[self.arm_joint_indices]

    def normalize_qpos(self, qpos: np.ndarray) -> torch.Tensor:
        """Normalize robot joint positions (radians -> normalized)"""
        qpos_remapped = self.remap_qpos_to_training_order(qpos)
        qpos_norm = (qpos_remapped - self.qpos_mean) / self.qpos_std
        return torch.from_numpy(qpos_norm).float().cuda().unsqueeze(0)

    def denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        """
        Denormalize model output to joint commands (normalized -> radians).
        Handles dimension mismatch if model outputs 16 dims but we need 14.
        """
        action_np = action.detach().cpu().numpy() # Shape: (1, Chunk, Dim) or (Dim,)

        # [Fix 1] Handle Output Dimension Mismatch
        # If model outputs 16 dims (14 joints + 2 dummy grippers), slice to 14 active joints.
        if action_np.shape[-1] == 16:
            action_np = action_np[..., :14]

        # Denormalize using 14-dim stats
        denorm = action_np * self.action_std + self.action_mean

        # Expand to 16-DoF commands for the robot
        expanded_shape = denorm.shape[:-1] + (16,)
        expanded = np.zeros(expanded_shape, dtype=denorm.dtype)
        # breakpoint()
        # Map 14 active joints to their physical indices (0-6, 8-14)
        expanded[..., self.arm_joint_indices] = denorm

        # Apply hardcoded gripper constraints
        expanded = self.apply_gripper_constraints(expanded)

        return expanded

    def apply_gripper_constraints(self, action: np.ndarray) -> np.ndarray:
        """Force gripper joints (7, 15) to commanded values."""
        if self.args.mode == "open_door":
            action[..., 7] = 0
            action[..., 15] = 0
        else:
            action[..., 7] = 1.0
            action[..., 15] = 1.0
        return action

    def preprocess_images(self, obs: dict) -> torch.Tensor:
        """Extract and preprocess camera images from observation"""
        curr_images = []

        for cam_name in self.camera_names:
            img = obs.get(cam_name)
            if img is None or (isinstance(img, np.ndarray) and img.size == 0):
                error_msg = (f"Failed to get a valid image from camera '{cam_name}'. ")
                print_color(error_msg, color="red", attrs=("bold",))
                raise ValueError(error_msg)

            # Ensure correct shape (H, W, 3) -> (3, H, W)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = np.transpose(img, (2, 0, 1))
            else:
                error_msg = (f"Invalid image shape: {img.shape}")
                print_color(error_msg, color="red", attrs=("bold",))
                raise ValueError(error_msg)

            curr_images.append(img)

        
        if self.black_image is not None:
            curr_images.append(self.black_image)
        
        
        # Stack and normalize
        image = np.stack(curr_images, axis=0)  # (num_cams, 3, H, W)
        image = torch.from_numpy(image / 255.0).float().cuda().unsqueeze(0)

        return image

    def warmup(self):
        """Warm up GPU and Torch Compiler"""
        print_color("\n[3/3] Warming up network...", color="cyan")

        obs = self.env.get_obs()
        qpos = self.normalize_qpos(obs['joint_positions'])
        image = self.preprocess_images(obs)

        # Run inference a few times
        with torch.inference_mode():
            for _ in range(5):
                _ = self.policy(qpos, image)

        print_color("  ✓ Network warmup complete", color="green")

    def run(self):
        """Main inference loop"""
        self.warmup()

        print_color("\n" + "="*70, color="cyan", attrs=("bold",))
        print_color("Starting Inference Loop", color="cyan", attrs=("bold",))
        print_color("="*70, color="cyan", attrs=("bold",))
        print_color("Press Ctrl+C to stop\n", color="yellow")

        # Setup query frequency and temporal aggregation
        query_frequency = self.args.query_frequency

        if self.args.temporal_agg:
            print_color(f"Temporal aggregation enabled with query_frequency: {query_frequency}", color="yellow")
            # Initialize all_time_actions buffer
            self.all_time_actions = torch.zeros([self.args.max_timesteps, self.args.max_timesteps + self.chunk_size, self.action_dim]).cuda()
            
            # [Optimization] Pre-compute exponential weights
            k = 0.01
            exp_weights_np = np.exp(-k * np.arange(self.chunk_size))
            exp_weights_np = exp_weights_np / exp_weights_np.sum()
            self.exp_weights = torch.from_numpy(exp_weights_np).cuda().unsqueeze(dim=1) # (chunk, 1)
        else:
            print_color(f"Using action buffer with query_frequency: {query_frequency}", color="yellow")

        # Performance tracking
        obs_times = []
        inf_times = []
        exec_times = []
        loop_times = []

        # --- Move to initial position safely ---
        print_color("Move to initial pose...", color='cyan')
        curr_joints = self.env.get_obs()['joint_positions']
        # Initial joint positions in actual robot order

        # switch off mode initial joints
        if self.args.mode == "switch_off":
            self.initial_joints = np.array([-0.20796557, -0.21495704, -0.14028409, -2.139557, 0.03464151, 2.0185945, -0.01454137, 1,
                                            0.08520385, 0.10141431, -0.00663875, -1.5505816, -0.02971161, 1.5962398, 0.14269795, 1])
        elif self.args.mode == "voltage_check":
            self.initial_joints = np.array([0.0325730, -0.2819586, -0.1306318, -1.9726636, -0.03116997, 1.7756970, 0.1323796, 1, 
                                            -0.0397095, -0.1140082, 0.0182835, -1.7956685, -0.0008271, 1.6466571, 0.0046001, 1])
        elif self.args.mode == "open_door":
            self.initial_joints = np.array([0.0363749, -0.1126141, -0.1155057, -1.8886392, -0.0725954, 1.8685882, 0.0151278, 1, 
                                            -0.0153577, -0.1184782, 0.0537092, -1.7913975, 0.0583810, 1.6946956, 0.0441786, 1])
        else:
            raise ValueError(f"Invalid mode: {self.args.mode}")
        
        abs_deltas = np.abs(self.initial_joints - curr_joints)
        steps = min(int(abs_deltas.max() / 0.01), 100)
        print_color(f"Moving to the initial position by {steps} steps...", color="cyan")

        # Linear interpolation to initial pose
        for jnt in np.linspace(curr_joints, self.initial_joints, steps):
            self.env.step(jnt)
            time.sleep(0.01)

        print_color("Initial position reached!", color="green")
        
        if self.args.mode == "switch_off":
            print_color("Switch_off mode: Joints 8-15 will be fixed to initial positions", color="yellow")
        
        start_time = time.time()
        dt = 1.0 / self.args.hz
        
        # [Fix 4] Soft Start Configuration
        # Blend from physical state to model state for the first N steps
        soft_start_steps = 20  # 0.4 seconds at 50Hz

        try:
            for t in range(self.args.max_timesteps):
                loop_start = time.time()

                # 1. Get observation
                obs_start = time.time()
                obs = self.env.get_obs()
                qpos_raw = obs['joint_positions']
                qpos = self.normalize_qpos(qpos_raw)
                image = self.preprocess_images(obs)
                obs_times.append(time.time() - obs_start)
                
                # 1.5. Visualize images
                if self.args.visualize and t % self.args.vis_frequency == 0:
                    self.visualize_images(obs, t)

                # 2. Run inference (every query_frequency steps)
                if t % query_frequency == 0:
                    inf_start = time.time()

                    with torch.inference_mode():
                        all_actions = self.policy(qpos, image)  # (1, chunk_size, Dim)

                        if all_actions.shape[-1] == 16:
                            all_actions = all_actions[..., :14]

                    inf_times.append(time.time() - inf_start)

                    if self.args.temporal_agg:
                        self.all_time_actions[[t], t:t+self.chunk_size] = all_actions
                    else:
                        # Denormalize entire chunk for standard action buffer
                        self.action_buffer = self.denormalize_action(all_actions)  # Returns (1, chunk, 16)
                        self.buffer_start_t = t

                # 3. Get action for current step
                if self.args.temporal_agg:
                    actions_for_curr_step = self.all_time_actions[:, t]  # (max_timesteps, action_dim)
                    actions_populated = torch.all(actions_for_curr_step != 0, dim=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    
                    if len(actions_for_curr_step) > 0:
                        # [Optimization] Use pre-computed weights
                        curr_weights = self.exp_weights[:len(actions_for_curr_step)]
                        curr_weights = curr_weights / curr_weights.sum()  # Re-normalize

                        # Weighted sum
                        raw_action = (actions_for_curr_step * curr_weights).sum(dim=0, keepdim=True)  # (1, action_dim)
                        action = self.denormalize_action(raw_action)[0]  # (16,)
                    else:
                        # No predictions yet, stay at current position
                        action = qpos_raw

                else:
                    if self.action_buffer is not None:
                        buffer_idx = t - self.buffer_start_t
                        if buffer_idx < self.action_buffer.shape[1]:
                            action = self.action_buffer[0, buffer_idx].copy()
                        else:
                            action = qpos_raw
                    else:
                        action = qpos_raw

                # 4. [Fix 4] Soft Start (Interpolation)
                # Prevents sudden jump at t=0 due to model/real mismatch
                abs_deltas = np.abs(action - qpos_raw)
                
                
                # time.sleep(0.05)
                if t < soft_start_steps:
                    alpha = (t + 1) / (soft_start_steps + 1)
                    action = (1 - alpha) * qpos_raw + alpha * action

                # # 5. Force gripper & mode constraints
                # action = self.apply_gripper_constraints(action)
                
                if self.args.mode == "switch_off":
                    action[8:15] = self.initial_joints[8:15]
                
                # 6. Execute action
                exec_start = time.time()
                self.env.step(action)
                # breakpoint()
                exec_times.append(time.time() - exec_start)

                # 7. Maintain control rate
                loop_time = time.time() - loop_start
                if loop_time < dt:
                    time.sleep(dt - loop_time)
                
                # time.sleep(0.05)
                total_loop_time = time.time() - loop_start
                loop_times.append(total_loop_time)

                # 8. Log progress
                if t % self.args.log_frequency == 0:
                    recent_loop = loop_times[-self.args.log_frequency:]
                    avg_loop = np.mean(recent_loop) * 1000
                    fps = 1000.0 / avg_loop if avg_loop > 0 else 0

                    elapsed = time.time() - start_time
                    print_color(
                        f"Step {t:4d}/{self.args.max_timesteps} | "
                        f"FPS: {fps:5.1f} | "
                        f"Loop: {avg_loop:5.1f}ms | "
                        f"Elapsed: {elapsed:.1f}s",
                        color="cyan"
                    )

        except KeyboardInterrupt:
            print_color("\n\nInterrupted by user (Ctrl+C)", color="yellow")

        except Exception as e:
            print_color(f"\n\nError during inference: {e}", color="red")
            traceback.print_exc()

        finally:
            # Close OpenCV windows
            if self.args.visualize:
                cv2.destroyAllWindows()
            self._cleanup(obs_times, inf_times, exec_times, loop_times)

    def _cleanup(self, obs_times, inf_times, exec_times, loop_times):
        """Save results and print summary"""
        print_color("\n" + "="*70, color="cyan")
        print_color("Shutting down...", color="cyan")
        print_color("="*70, color="cyan")

        # Print performance summary
        if len(loop_times) > 0:
            print_color("\nPerformance Summary:", color="cyan", attrs=("bold",))
            print_color(f"  Total steps: {len(loop_times)}", color="white")
            print_color(f"  Average FPS: {1.0 / np.mean(loop_times):.2f}", color="white")
            print_color(f"  Observation time: {np.mean(obs_times)*1000:.2f}ms", color="white")
            if len(inf_times) > 0:
                print_color(f"  Inference time: {np.mean(inf_times)*1000:.2f}ms", color="white")
                print_color(f"  Inference frequency: every {len(loop_times) / len(inf_times):.1f} steps", color="white")
            print_color(f"  Execution time: {np.mean(exec_times)*1000:.2f}ms", color="white")
            print_color(f"  Loop time: {np.mean(loop_times)*1000:.2f}ms", color="white")

        print_color(f"\n✓ All outputs saved to: {self.output_dir}", color="green", attrs=("bold",))
        print_color("="*70, color="cyan")
        print_color("Shutdown complete", color="green", attrs=("bold",))
        print_color("="*70, color="cyan")


def main(args: PandaInferenceArgs):
    """Main entry point"""
    engine = PandaInferenceEngine(args)
    engine.run()


if __name__ == "__main__":
    main(tyro.cli(PandaInferenceArgs))