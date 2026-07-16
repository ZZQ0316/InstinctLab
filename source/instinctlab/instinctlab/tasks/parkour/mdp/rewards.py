from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(env, command_name: str, vel_threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    # no reward for zero command
    reward *= torch.logical_or(
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > vel_threshold,
        torch.abs(env.command_manager.get_command(command_name)[:, 2]) > vel_threshold,
    )
    return reward


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.15,
    offset: float = 1.0,
) -> torch.Tensor:
    """Penalize moving when there is no velocity command."""
    asset = env.scene[asset_cfg.name]
    dof_error = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return (
        (dof_error - offset)
        * (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < threshold)
        * (torch.abs(env.command_manager.get_command(command_name)[:, 2]) < threshold)
    )


def feet_close_xy_gauss(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 0.1
) -> torch.Tensor:
    """Penalize when feet are too close together in the y distance."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]

    # Get feet positions (assuming first two body_ids are left and right feet)
    left_foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids[0], :2]
    right_foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids[1], :2]
    heading_w = asset.data.heading_w

    # Transform feet positions to robot frame
    cos_heading = torch.cos(heading_w)
    sin_heading = torch.sin(heading_w)

    # Rotate to robot frame
    left_foot_robot_frame = torch.stack(
        [
            cos_heading * left_foot_xy[:, 0] + sin_heading * left_foot_xy[:, 1],
            -sin_heading * left_foot_xy[:, 0] + cos_heading * left_foot_xy[:, 1],
        ],
        dim=1,
    )

    right_foot_robot_frame = torch.stack(
        [
            cos_heading * right_foot_xy[:, 0] + sin_heading * right_foot_xy[:, 1],
            -sin_heading * right_foot_xy[:, 0] + cos_heading * right_foot_xy[:, 1],
        ],
        dim=1,
    )

    feet_distance_y = torch.abs(left_foot_robot_frame[:, 1] - right_foot_robot_frame[:, 1])

    # Return continuous penalty using exponential decay
    return torch.exp(-torch.clamp(threshold - feet_distance_y, min=0.0) / std**2) - 1


def heading_error(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Compute the heading error between the robot's current heading and the goal heading."""
    # compute the error
    ang_vel_cmd = torch.abs(env.command_manager.get_command(command_name)[:, 2])
    return ang_vel_cmd


def dont_wait(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize standing still when there is a forward velocity command."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_cmd_x = env.command_manager.get_command(command_name)[:, 0]
    lin_vel_x = asset.data.root_lin_vel_b[:, 0]
    return (lin_vel_cmd_x > 0.3) * ((lin_vel_x < 0.15).float() + (lin_vel_x < 0).float() + (lin_vel_x < -0.15).float())


def feet_orientation_contact(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward feet being oriented vertically when in contact with the ground."""
    # 惩罚脚触地时不水平，鼓励脚在触地时保持水平，避免脚尖或脚跟着地。
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    left_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    left_projected_gravity = quat_apply_inverse(left_quat, asset.data.GRAVITY_VEC_W)
    right_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[1], :]
    right_projected_gravity = quat_apply_inverse(right_quat, asset.data.GRAVITY_VEC_W)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1

    return (
        torch.sum(torch.square(left_projected_gravity[:, :2]), dim=-1) ** 0.5 * is_contact[:, 0]
        + torch.sum(torch.square(right_projected_gravity[:, :2]), dim=-1) ** 0.5 * is_contact[:, 1]
    )


def feet_at_plane(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    left_height_scanner_cfg: SceneEntityCfg,
    right_height_scanner_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_offset=0.035, # 期望的脚离地高度
) -> torch.Tensor:
    """Reward feet being at certain height above the ground plane."""
    # 配合负权重是一个惩罚项，用来禁止脚在触地时抬得太高（超过3.5cm就开始惩罚）
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # 如果最大接触力 > 1N，则认为脚触地了
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, contact_sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1
    # 获取地面高度
    left_sensor = env.scene[left_height_scanner_cfg.name]
    left_sensor_data = left_sensor.data.ray_hits_w[..., 2]
    left_sensor_data = torch.where(torch.isinf(left_sensor_data), 0.0, left_sensor_data)
    right_sensor = env.scene[right_height_scanner_cfg.name]
    right_sensor_data = right_sensor.data.ray_hits_w[..., 2]
    right_sensor_data = torch.where(torch.isinf(right_sensor_data), 0.0, right_sensor_data)
    # 获取脚的实际高度
    left_height = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    right_height = asset.data.body_pos_w[:, asset_cfg.body_ids[1], 2]

    left_reward = (
        torch.clamp(left_height.unsqueeze(-1) - left_sensor_data - height_offset, min=0.0, max=0.3) * is_contact[:, 0:1]
    )
    right_reward = (
        torch.clamp(right_height.unsqueeze(-1) - right_sensor_data - height_offset, min=0.0, max=0.3)
        * is_contact[:, 1:2]
    )
    return torch.sum(left_reward, dim=-1) + torch.sum(right_reward, dim=-1)


def link_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat link orientation using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    link_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    link_projected_gravity = quat_apply_inverse(link_quat, asset.data.GRAVITY_VEC_W)

    return torch.sum(torch.square(link_projected_gravity[:, :2]), dim=1)

# 从github上找的一个二阶动作平滑的奖励函数，鼓励动作加速度小，适合需要平滑动作的任务，比如跑酷。
def action_acc_l2(
    env: ManagerBasedRLEnv,
    action_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Second-order action smoothness (action acceleration penalty).
    
    Penalizes: -∑ⱼ(aⱼ,ₜ − 2aⱼ,ₜ₋₁ + aⱼ,ₜ₋₂)²
    """
    action = env.action_manager.action
    if action_ids is not None:
        action = action[:, action_ids]
    # Compute the second-order finite difference (acceleration)
    if not hasattr(env, "_last_action"):
        env._last_action = torch.zeros_like(action)
        env._last_last_action = torch.zeros_like(action)
    action_acc = action - 2 * env._last_action + env._last_last_action
    # Update the last actions
    env._last_last_action[:] = env._last_action
    env._last_action[:] = action
    # Compute the L2 penalty
    action_acc_l2 = torch.sum(torch.square(action_acc), dim=-1)  # (batch_size,)
    return action_acc_l2


def feet_contact_area(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    left_area_scanner_cfg: SceneEntityCfg,
    right_area_scanner_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sole_offset: float = 0.035,
    height_tolerance: float = 0.0,
    min_ratio: float = 0.85,
) -> torch.Tensor:
    """奖励触地时具有足够支撑面积的脚底。

    通过脚底 raycast 网格估计每只脚的支撑面积比例。
    当脚处于接触状态时：
    - 支撑比例低于 ``min_ratio`` 时，奖励为 0
    - 支撑比例位于 ``min_ratio`` 到 1.0 之间时，奖励线性映射到 [0, 1]
    ``sole_offset`` 表示 ankle_roll_link 原点到脚底面的竖直距离。
    """
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    is_contact = (
        contact_sensor.data.net_forces_w_history[:, :, contact_sensor_cfg.body_ids]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )

    left_scanner = env.scene[left_area_scanner_cfg.name]
    left_ground_z = left_scanner.data.ray_hits_w[..., 2]
    left_ground_z = torch.where(torch.isinf(left_ground_z), 0.0, left_ground_z)
    right_scanner = env.scene[right_area_scanner_cfg.name]
    right_ground_z = right_scanner.data.ray_hits_w[..., 2]
    right_ground_z = torch.where(torch.isinf(right_ground_z), 0.0, right_ground_z)

    left_foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2:3]
    right_foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids[1], 2:3]

    # 扣除 sole_offset 后，平放时 foot_z - ground_z ≈ 0
    left_gap = (left_foot_z - sole_offset) - left_ground_z
    right_gap = (right_foot_z - sole_offset) - right_ground_z

    left_supported = (left_gap.abs() < height_tolerance).float()
    right_supported = (right_gap.abs() < height_tolerance).float()

    left_ratio = left_supported.mean(dim=-1)
    right_ratio = right_supported.mean(dim=-1)

    # if env.common_step_counter % 50 == 0:
    #     print(f"[feet_contact_area] L_ratio={left_ratio[0].item():.2f} R_ratio={right_ratio[0].item():.2f} "
    #           f"L_gap_mean={left_gap[0].mean().item():.4f} R_gap_mean={right_gap[0].mean().item():.4f}")

    left_reward = torch.clamp(left_ratio - min_ratio, min=0.0) / (1.0 - min_ratio) * is_contact[:, 0]
    right_reward = torch.clamp(right_ratio - min_ratio, min=0.0) / (1.0 - min_ratio) * is_contact[:, 1]

    return left_reward + right_reward