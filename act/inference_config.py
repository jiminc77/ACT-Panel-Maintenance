from dataclasses import dataclass
from typing import List, Dict, Any
import pickle
import os


@dataclass
class InferenceConfig:
    """Configuration for Mobile ALOHA inference"""

    # Inference mode
    inference_mode: str = 'voltage_check'  # 'voltage_check' or 'switch_off'

    # Checkpoint related
    ckpt_dir: str = None  # If None, auto-select based on inference_mode
    ckpt_name: str = 'policy_best.ckpt'

    # Execution settings
    max_timesteps: int = 1000
    temporal_agg: bool = False

    # Debug settings
    debug: bool = False
    log_frequency: int = 10
    save_trajectory: bool = True

    # Fixed joint positions for switch_off mode (joints 7-12, right arm)
    # Median values from GT data analysis
    fixed_right_arm_joints: List[float] = None

    # Fixed gripper positions (joints 6 and 13)
    fixed_gripper_left: float = 0.0  # Joint 6
    fixed_gripper_right: float = 0.0  # Joint 13

    # Derived attributes (populated in __post_init__)
    ckpt_path: str = None
    policy_config_path: str = None
    stats_path: str = None
    policy_config: Dict[str, Any] = None
    stats: Dict[str, Any] = None
    camera_names: List[str] = None
    chunk_size: int = None
    query_frequency: int = None

    def __post_init__(self):
        """Validate paths and load configuration files"""
        # Validate inference mode
        if self.inference_mode not in ['voltage_check', 'switch_off']:
            raise ValueError(f"Invalid inference_mode: {self.inference_mode}. Must be 'voltage_check' or 'switch_off'")

        # Auto-select checkpoint directory based on inference mode if not specified
        if self.ckpt_dir is None:
            self.ckpt_dir = f"./ckpts/{self.inference_mode}"
            print(f"Auto-selected checkpoint directory: {self.ckpt_dir}")

        # Checkpoint path
        self.ckpt_path = os.path.join(self.ckpt_dir, self.ckpt_name)
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_path}")

        # Config file path
        self.policy_config_path = os.path.join(self.ckpt_dir, 'config.pkl')
        if not os.path.exists(self.policy_config_path):
            raise FileNotFoundError(f"Config not found: {self.policy_config_path}")

        with open(self.policy_config_path, 'rb') as f:
            self.policy_config = pickle.load(f)

        # Stats file path
        self.stats_path = os.path.join(self.ckpt_dir, 'dataset_stats.pkl')
        if not os.path.exists(self.stats_path):
            raise FileNotFoundError(f"Stats not found: {self.stats_path}")

        with open(self.stats_path, 'rb') as f:
            self.stats = pickle.load(f)

        # Extract settings
        self.camera_names = self.policy_config['camera_names']
        self.chunk_size = self.policy_config['policy_config']['num_queries']
        self.query_frequency = 1 if self.temporal_agg else self.chunk_size

        # Set fixed right arm joint positions for switch_off mode
        # Median values from switch_off dataset (all episodes combined, IQR filtered)
        if self.fixed_right_arm_joints is None:
            self.fixed_right_arm_joints = [
                -9.220000,   # Joint 7
                17.049999,   # Joint 8
                40.950001,   # Joint 9
                -40.250000,  # Joint 10
                67.849998,   # Joint 11
                29.610001    # Joint 12
            ]

    def summary(self) -> str:
        """Return configuration summary as string"""
        lines = [
            "="*60,
            "Inference Configuration Summary",
            "="*60,
            f"Inference mode: {self.inference_mode}",
            f"Checkpoint directory: {self.ckpt_dir}",
            f"Checkpoint file: {self.ckpt_name}",
            f"Cameras: {', '.join(self.camera_names)}",
            f"Chunk size: {self.chunk_size}",
            f"Query frequency: {self.query_frequency}",
            f"Temporal aggregation: {self.temporal_agg}",
            f"Max timesteps: {self.max_timesteps}",
            f"Debug mode: {self.debug}",
        ]

        if self.inference_mode == 'switch_off':
            lines.append(f"Fixed right arm joints: {self.fixed_right_arm_joints}")

        lines.append("="*60)
        return "\n".join(lines)
