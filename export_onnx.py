#!/usr/bin/env python3
"""Script to export RL policy to ONNX model without running simulation."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "scripts", "instinct_rl"))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Export RL policy to ONNX model.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# custom play arguments
parser.add_argument("--env_cfg", action="store_true", default=False, help="Load configuration from file.")
parser.add_argument("--agent_cfg", action="store_true", default=False, help="Load configuration from file.")

# append Instinct-RL cli arguments
cli_args.add_instinct_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# set headless mode by default
if not hasattr(args_cli, "headless") or not args_cli.headless:
    args_cli.headless = True
# disable cameras
args_cli.enable_cameras = False

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from instinct_rl.runners import OnPolicyRunner
from instinct_rl.utils.utils import get_subobs_by_components, get_subobs_size

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import load_pickle, load_yaml
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg


def main():
    """Export policy to ONNX."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: InstinctRlOnPolicyRunnerCfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "instinct_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    agent_cfg.load_run = args_cli.load_run
    if agent_cfg.load_run is not None:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if os.path.isabs(agent_cfg.load_run):
            resume_path = get_checkpoint_path(
                os.path.dirname(agent_cfg.load_run), os.path.basename(agent_cfg.load_run), agent_cfg.load_checkpoint
            )
            log_dir = os.path.dirname(resume_path)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            log_dir = os.path.dirname(resume_path)
    else:
        raise RuntimeError("Please specify a checkpoint to load with --load_run.")

    print(f"[INFO] Log directory: {log_dir}")

    if args_cli.env_cfg:
        env_cfg = load_pickle(os.path.join(log_dir, "params", "env.pkl"))
    if args_cli.agent_cfg:
        agent_cfg_dict = load_yaml(os.path.join(log_dir, "params", "agent.yaml"))
    else:
        agent_cfg_dict = agent_cfg.to_dict()

    # set single environment
    env_cfg.scene.num_envs = 1
    # disable virtual obstacles visualization if needed
    if hasattr(env_cfg.scene.terrain, "debug_vis"):
        env_cfg.scene.terrain.debug_vis = False

    # create isaac environment (no rendering)
    env = gym.make(args_cli.task, cfg=env_cfg)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for instinct-rl
    env = InstinctRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    if agent_cfg.load_run is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        ppo_runner.load(resume_path)

    # export policy to ONNX
    export_model_dir = os.path.join(log_dir, "exported")
    print(f"[INFO]: Exporting ONNX model to: {export_model_dir}")
    if not os.path.exists(export_model_dir):
        os.makedirs(export_model_dir, exist_ok=True)

    # get one observation to trace ONNX export
    obs, _ = env.get_observations()
    ppo_runner.alg.actor_critic.export_as_onnx(obs, export_model_dir)

    print(f"[INFO]: ONNX model exported successfully!")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
