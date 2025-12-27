from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Visualize HDF5 episode contents.")
    parser.add_argument(
        "hdf5_path",
        nargs="?",
        default="/home/demo-panda/workspace/gello_software/etri_switchoff/act/episode_0.hdf5",
        help="Episode HDF5 file to inspect.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=4,
        help="Number of sample frames to visualize per camera.",
    )
    args = parser.parse_args()

    hdf5_path = Path(args.hdf5_path).expanduser()
    with h5py.File(hdf5_path, "r") as h5f:
        wrist1 = np.asarray(h5f["observations/images/wrist1"])[: args.num_images]
        wrist2 = np.asarray(h5f["observations/images/wrist2"])[: args.num_images]
        qpos = np.asarray(h5f["observations/qpos"])
        action = np.asarray(h5f["action"])

    # plot images
    cols = args.num_images
    fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 6))
    for idx in range(cols):
        axes[0, idx].imshow(wrist1[idx])
        axes[0, idx].axis("off")
        axes[0, idx].set_title(f"wrist1 frame {idx}")
        axes[1, idx].imshow(wrist2[idx])
        axes[1, idx].axis("off")
        axes[1, idx].set_title(f"wrist2 frame {idx}")
    fig.suptitle("Sample wrist images")
    plt.tight_layout()
    plt.savefig("hdf5_images.png")
    plt.close(fig)

    # plot qpos/action
    timesteps = np.arange(qpos.shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(timesteps, qpos)
    axes[0].set_ylabel("qpos values")
    axes[0].legend([f"joint {i}" for i in range(qpos.shape[1])], ncol=4, fontsize=6)
    axes[1].plot(timesteps, action)
    axes[1].set_ylabel("action values")
    axes[1].set_xlabel("timestep")
    axes[1].legend([f"action {i}" for i in range(action.shape[1])], ncol=4, fontsize=6)
    fig.suptitle("Qpos and Action over time")
    plt.tight_layout()
    plt.savefig("hdf5_states.png")
    plt.close(fig)

    print("saved to hdf5_images.png and hdf5_states.png")


if __name__ == "__main__":
    main()