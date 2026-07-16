# Context

当前用户要先为 parkour 管线加入**真实 CoP（Center of Pressure）计算**，要求尽量最小侵入，不先做 CoP-based reward，只先把 CoP 作为可读取的传感量接进现有 scene/sensor 管线。现有 [source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py](source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py) 里使用的是 IsaacLab 自带 `ContactSensorCfg`，而该路径最终只暴露 `net_forces_w / net_forces_w_history / force_matrix_w`，不提供计算真实 CoP 所需的逐接触点位置。

已验证：底层 Isaac Sim / PhysX tensor API 的 `RigidContactView.get_contact_data(dt)` 可提供逐接触点的 `contact points / normals / normal force magnitudes / separation distances`，因此**当前环境底层有能力真算 CoP**；缺的是项目当前没有一条最小化的接入通路。

为了避免修改外部 IsaacLab `ContactSensor` 实现、降低 blast radius，推荐新增一个项目内的并行专用传感器，仅为双脚提供 CoP 数据，并像现有 `VolumePoints` 一样通过 `env.scene.sensors[...]` 访问。

# Recommended approach

## 1. 新增项目内 CoP 专用传感器，而不是修改 IsaacLab ContactSensor

在 [source/instinctlab/instinctlab/sensors/](source/instinctlab/instinctlab/sensors/) 下新增一个小型传感器模块，例如：
- `contact_cop/contact_cop.py`
- `contact_cop/contact_cop_cfg.py`
- `contact_cop/contact_cop_data.py`

实现方式直接复用项目里已有自定义 sensor 模式，参考：
- [source/instinctlab/instinctlab/sensors/volume_points/volume_points.py](source/instinctlab/instinctlab/sensors/volume_points/volume_points.py)
- [source/instinctlab/instinctlab/sensors/volume_points/volume_points_cfg.py](source/instinctlab/instinctlab/sensors/volume_points/volume_points_cfg.py)
- [source/instinctlab/instinctlab/sensors/volume_points/volume_points_data.py](source/instinctlab/instinctlab/sensors/volume_points/volume_points_data.py)

原因：
- 不修改外部依赖 `IsaacLab/source/isaaclab/.../contact_sensor.py`
- 不影响当前所有 `contact_forces` 消费者
- 仍然保持现有 `scene.sensors` 的使用习惯，方便后续 reward 直接接入

## 2. 传感器初始化时单独创建 rigid contact view，并开启 detailed contact data buffer

在新传感器的 `_initialize_impl()` 中：
- 像 `ContactSensor` 一样创建 `RigidBodyView` / `RigidContactView`
- 但必须在创建 contact view 时启用 detailed contact data 所需的 `max_contact_data_count > 0`
- 过滤对象只针对地面/台阶相关 mesh，避免收集到无关接触

需要参考的底层接口来源：
- IsaacLab contact sensor 如何创建 view：
  [contact_sensor.py:278-282](../../../../IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py#L278-L282)
- PhysX tensor API 提供 detailed contact 数据：
  `/home/tcict/anaconda3/envs/isaaclab_450/lib/python3.10/site-packages/isaacsim/extsPhysics/omni.physics.tensors/omni/physics/tensors/impl/api.py`
  - `SimulationView.create_rigid_contact_view(..., max_contact_data_count=0)`
  - `RigidContactView.get_contact_data(dt)`

这里的关键是：**不要复用当前 `contact_forces` 实例**，而是在项目内新建一条并行数据通路，只为 CoP 服务。

## 3. 定义一个最小数据结构，只暴露当前阶段真正需要的 CoP 结果

在 `contact_cop_data.py` 中新增 dataclass，例如包含：
- `cop_w: torch.Tensor`，shape `(N, B, 3)`，每个 env、每只脚的世界坐标 CoP
- `in_contact: torch.Tensor`，shape `(N, B)`，该脚本步是否存在有效接触点
- `normal_force_scalar: torch.Tensor` 或 `normal_force_w: torch.Tensor`
- `num_contact_points: torch.Tensor`，shape `(N, B)`，便于调试

如果实现成本低，可额外加：
- `cop_b: torch.Tensor`，body frame 下的 CoP，后续写 reward 更方便

当前阶段**不需要**：
- contact history
- friction-based扩展量
- 通用全身所有 link 的大而全接口
- 复杂 debug visualization

目标是先把 CoP 可靠算出来、可打印、可在 reward 中读取。

## 4. 在 `_update_buffers_impl()` 中调用 detailed contact API 并计算每只脚的真实 CoP

每步更新时：
1. 调用 `RigidContactView.get_contact_data(dt)` 读取扁平 contact buffers
2. 使用返回的 `pair_contact_counts / pair_contact_start_indices` 将 contact stream 映射回 `(env, foot)`
3. 对每只脚的所有有效 contact point：
   - 读取 `contact_point_i`
   - 读取 `normal_i`
   - 读取 `normal_force_i`
   - 构造法向力向量 `F_i = normal_force_i * normal_i`
4. 用法向力标量作为权重计算 CoP：
   - `cop = sum(point_i * normal_force_i) / sum(normal_force_i)`
5. 如果该脚没有有效接触，设置：
   - `in_contact=False`
   - `cop_w=0` 或保持约定的无效值

注意：这一版只做**法向 CoP**，不先引入 friction data。这样实现最简单，也和标准足底压力中心定义一致。

## 5. 将新传感器导出并注册到 parkour scene 中，但先不改 reward 逻辑

需要：
- 在 [source/instinctlab/instinctlab/sensors/__init__.py](source/instinctlab/instinctlab/sensors/__init__.py) 中导出新传感器类型
- 在 [source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py](source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py) 的 scene 部分新增一个 `ContactCoPCfg`（或类似命名）
- `prim_path` 直接对准双脚：`{ENV_REGEX_NS}/Robot/.*_ankle_roll_link`
- `update_period` 与 sim dt 保持一致，和现有 `contact_forces` 同步
- `filter_prim_paths_expr` 只指向地形/ground mesh，确保 CoP 只来自脚-地形接触

当前阶段不要把它接进任何 reward，只需要让它能在 play/train 中被读取和打印。

## 6. 预留一个最小验证入口，而不是立即加 reward

第一步验证建议放在 reward/调试代码里临时打印以下量：
- env0 左右脚 `in_contact`
- `num_contact_points`
- `cop_w`
- 总法向力

验证通过后，再决定是否加以下 reward：
- CoP 距离脚底边缘的 margin reward
- CoP 距离台阶前沿安全区的 reward
- CoP 平滑性 reward

但这些都不属于当前这一步。

# Critical files

- [source/instinctlab/instinctlab/sensors/__init__.py](source/instinctlab/instinctlab/sensors/__init__.py)
- 新增：`source/instinctlab/instinctlab/sensors/contact_cop/contact_cop.py`
- 新增：`source/instinctlab/instinctlab/sensors/contact_cop/contact_cop_cfg.py`
- 新增：`source/instinctlab/instinctlab/sensors/contact_cop/contact_cop_data.py`
- [source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py](source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py)

# Existing code to reuse

- 自定义 sensor 模式：
  - [source/instinctlab/instinctlab/sensors/volume_points/volume_points.py](source/instinctlab/instinctlab/sensors/volume_points/volume_points.py)
  - [source/instinctlab/instinctlab/sensors/volume_points/volume_points_cfg.py](source/instinctlab/instinctlab/sensors/volume_points/volume_points_cfg.py)
  - [source/instinctlab/instinctlab/sensors/volume_points/volume_points_data.py](source/instinctlab/instinctlab/sensors/volume_points/volume_points_data.py)
- IsaacLab ContactSensor 如何创建 rigid contact view：
  - [contact_sensor.py:278-343](../../../../IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py#L278-L343)
- 当前 parkour 如何消费 sensor：
  - [source/instinctlab/instinctlab/tasks/parkour/mdp/rewards.py](source/instinctlab/instinctlab/tasks/parkour/mdp/rewards.py)
- 当前 parkour 如何注册 sensor：
  - [source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py](source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py)

# Verification

1. **静态检查**
   - 新传感器模块可被 `instinctlab.sensors` 正常导入
   - `parkour_env_cfg.py` 中新 sensor 配置能被构造

2. **运行时检查**
   - play 启动时新 sensor 成功初始化，无 `max_contact_data_count` / filter 配置错误
   - 平站时左右脚 `in_contact=True` 且 `num_contact_points > 0`

3. **数值检查**
   - 左右脚 CoP 坐标随脚位置移动而同步变化
   - 平地双脚静站时 CoP 位于脚掌支撑区域内部，而不是离散跳变到异常位置
   - 单脚支撑时，对应脚仍能稳定输出 CoP，另一脚 `in_contact=False`

4. **边界检查**
   - 脚离地时不会产生 NaN（无接触时安全回退）
   - 与现有 `contact_forces`、`feet_at_plane`、`feet_contact_area` 共存时不改变它们原有行为

5. **性能/容量检查**
   - 若 detailed contact buffer 不足，按需为 parkour sim 增加 PhysX rigid contact capacity 配置
   - 确认只对双脚启用该 sensor，避免不必要的全身接触数据开销
