## 动作

**目标关节位置** = default_joint_pos + 策略输出 * scale
- 策略输出被限制在 [-1, 1] 时，实际目标位置在 default_joint_pos ± scale 范围内。
- 合理性：策略输出0时，机器人目标状态是默认的安全状态

## 速度指令生成机制

- 每隔resampling__time_range=(8.0, 12.0)
  - 重新选取目标位置
  - 在当前的地形允许速度范围内，随机生成最大线速度/角速度阈值
  - 以一定概率rel_standing_envs=0.05将环境设置为静止模式
- 在每个控制步长（50HZ=0.02s）
  - 实时计算当前与目标位置的距离
  - 生成线速度与角速度指令：速度=误差*stifness(2.0)
  - 用速度阈值对速度指令进行裁剪
  - 如果当前env是静止模式，则清除速度指令

----------------------------------------------------

**play键盘控制测试时，键盘直接提供速度指令观测，没有上述基于距离的动态计算处理**
- w/Q - 前进加速 步长0.1
- F/G - 左转/右转 1.0/-1.0 单值映射
- S - 停止旋转
- X - 清空速度指令

## 关于训练

地形配置等相关环境参数都保存在训练日志文件中的params/env.yaml中，不用手工记录

## 当前parkour策略网络

- 当前parkour任务使用的是 **Encoder + MoE Actor-Critic**
- policy配置在 `source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py`
- `MoEPolicyCfg` 继承 `InstinctRlEncoderMoEActorCriticCfg`
- 专家数 `num_moe_experts = 4`
- actor隐藏层: `[256, 128, 64]`
- critic隐藏层: `[256, 128, 64]`
- 深度编码器 `DepthEncoderConv2dCfg`:
  - `output_size = 128`
  - `channels = [4]`
  - `kernel_sizes = [3]`
  - `strides = [1]`
  - `hidden_sizes = [256, 256]`
  - 输入组件: `depth_image`

## 当前parkour推理机制

- 推理时先输入原始观测（含depth等分段观测）
- 然后先经过 `encoders(observations)`，把观测编码成embedding
- 再把embedding送入 MoE actor
- MoE actor内部包含：
  - 1个 gate 网络
  - 4个 expert MLP
- gate输出4个权重，并经过softmax归一化
- 4个expert同时对同一个embedding做前向计算
- 最终动作输出 = 4个expert输出按gate权重加权求和
- 数学形式： `a = Σ w_i(x) * expert_i(x)`
- 其中 `w_i(x)` 是 gate 根据当前输入计算出的权重，4个权重和为1
- 当前 parkour 配置里没有单独设置 `moe_gate_hidden_dims`
  - 因此 gate 实际上是 **单层线性层 + softmax**
- 推理模式下不采样，直接输出 actor 的均值动作
- 然后动作再经过：
  - `目标关节位置 = default_joint_pos + 策略输出 * scale`
  - 再由底层 delayed PD actuator 转成关节控制

## 脚与虚拟障碍穿透惩罚触发条件

- 奖励项是 `volume_points_penetration`
- 检测对象是双脚 `ankle_roll_link` 附近的体积点
  - 每只脚 10×5×2 个点
  - 覆盖范围约为 x∈[-2.5cm, 12cm], y∈[-3cm, 3cm], z∈[-4cm, 0cm]
- 虚拟障碍是台阶前沿凸边缘处的圆柱体
  - `convex_edges_only=True`
  - 圆柱半径 2.5cm
  - 相对边缘向外偏移 2.5cm，向上偏移 2.5cm
- 单个体积点满足以下条件时记为穿透：
  - 点在线段圆柱体的轴向投影范围内
  - 点到圆柱轴线距离 `< 半径`
- 穿透深度 = `半径 - 点到轴线距离`
- 奖励项不是单纯统计是否穿透，而是按以下量累加：
  - `Σ(是否穿透 * 点速度 * 穿透深度)`
- 因此：
  - 只要体积点进入虚拟圆柱体内部，就会判定为穿透
  - 点速度越大、穿透越深，惩罚越大
  - 即使发生穿透，如果该点几乎静止，惩罚也会很小

## 为什么isaac中play的策略良好，但mujoco中sim2sim的效果很差？

- 速度机制的原因吗？
  不是！因为上面提到了，play时速度指令并不是基于距离实时计算的；
- 机器人本身配置（urdf）有差距吗？
  好像也不是，之前有系统对比过两个urdf，基本没有太大差别；
- 会是深度信息的差异吗？
  很可能是，但暂时找不出原因；

## 深度相机配置

- 挂载在toso link上，仅随yaw对齐
- 位姿：
  - pos=(0.0487988662332928, 0.01, 0.4378029937970051)
  - rot_wxyz=(0.9135367613482678, 0.004363309284746571, 0.4067366430758002, 0.0)
- 经计算相机主轴与重力方向夹角为42度，与实机一致
- 分辨率：(width, height)=(64, 36) fovx=89.51 fovy=58.29
- 裁剪距离: (0.1, 2.5)
- 噪声处理：
  - crop_range: (18, 0, 16, 16)，最终得到18*32大小的深度图
  - 高斯噪声：深度范围内添加std=0.03的高斯噪声 no
  - 深度伪影：以0.0001的概率注入  no 
  - 高斯模糊：3*3
  - OOD Perturbation no
  - 归一化：把深度裁到[0, 2.5]并归一化到[0, 1]
- 历史缓存：
  - 历史队列长37
  - 每跳5帧取1帧，共8帧

## 问题：

- 参考动作选择仅发生在仿真startup、环境reset、动作exhausted时
- 所有env无论地形和速度，随机从混杂的单个参考动作中截取片段作为参考

**修改**
- stand_still.npz 这个参考动作只作用于静止模式的env
- plane_walk.npz 只作用于perlin_rough 与 hf_pyramid_slope_inv 地形中的非静止env
- stairs_up.npz 只作用于 pyramid_stairs 地形的非静止env
- stairs_down.npz 只作用于 pyramid_stairs_inv 地形的非静止env
- 如果存在角速度，则参考运动选择turn_left.npz、turn_right.npz

## 为什么下楼表现这么差？

感觉视觉偏差只是一方面
主要下楼时即便一直踩边缘，最终也很难摔倒，因为重心前倾还是和前进方向一致
但上楼时踩边缘导致重心偏向与前进方向相反的后方

## 学到了

对于机器人碰到障碍后通过“绕开”而非“跨越”来获得高奖励时
本质上是奖励函数存在漏洞，这个问题也叫做 reward hacking
而InstinctLab采用的“目标区域”采样策略正是为了解决这个问题

## 虚拟障碍尺寸与偏移的待解决问题

- 当前脚部虚拟障碍用固定圆柱半径与固定偏移表示台阶前沿约束
- 对于最小深度约 21cm 的台阶，当前配置基本合适
- 但当台阶深度明显大于 21cm（如 25cm、30cm）时，仅使用当前固定配置不足以同时满足两个目标：
  - 台阶外侧仍需保持稳定的惩罚覆盖，避免脚越出台阶边缘
  - 台阶内侧也应根据额外深度形成一定缓冲区，约束机器人不要贴着边缘落脚
- 只把圆柱整体内移会削弱外侧惩罚
- 更合理的方向是：外侧覆盖保持固定，内侧覆盖随台阶额外深度自适应增加，同时联动调整圆柱半径与偏移
- 一个待验证的设计思路：
  - 外侧固定保持 5cm 障碍覆盖
  - 22cm 及以下深度台阶，内侧不额外设障碍，即保持当前默认实现
  - 22cm 以上深度台阶，内侧覆盖按 `台阶深度 - 22cm` 增加
  - 例如 25cm 台阶：内侧覆盖 3cm，配合外侧固定 5cm，总圆柱直径为 8cm
- 该问题暂未实现，后续如继续处理，需要把“外侧固定覆盖 + 内侧随台阶深度自适应覆盖”的规则落实到虚拟圆柱生成逻辑中

## 2026-06-24 虚拟障碍改动记录

- terrain generator 改为保留每个 sub-terrain 的 `terrain_name`，并把 terrain context 传给 virtual obstacle 生成逻辑
- edge cylinder 支持按 terrain name 覆盖参数
- `pyramid_stairs_small` 继续使用小圆柱：半径 2.5cm，外偏 2.5cm，上偏 2.5cm，且只对凸边缘生效
- 其他 terrain：
  - 凸边缘：大圆柱，半径 3.5cm，外偏 1.5cm，上偏 3.5cm
  - 凹边缘：也加虚拟障碍，半径 2cm，仅向上偏移2cm
- `pose_velocity_command.py` 同时放宽了 terrain name 检查：`velocity_ranges` 里存在但当前 play 场景未生成的 terrain type 会直接跳过，不再报错

## 关于部署
- 自己训的模型，由于depth_noise_pipeline并没有与作者官方模型对齐（不可知）
- 直接部署会不会出现机器人失控？

## 当前与落脚相关的奖励
- volume_points_penetration — 脚体积点穿入虚拟圆柱的惩罚（速度×穿透深度），无触发条件
- step_safety（未启用）— 奖励在无穿透时触地：-log(max_penetration) * is_contact
- feet_air_time — 奖励单脚站立/迈步，且迈步持续时间越长越好
- feet_slide — 惩罚接触时的脚底滑动（XY线速度）
- feet_flat_ori — 惩罚接触时脚底不平（非水平）
- feet_at_plane — 惩罚触地时脚底高于台面3.5cm及以上的情况
补充
- feet_contact_area — 线性奖励大于等于80% 的落脚比例
- link_orientation已经在惩罚重心偏移了

## 虚拟障碍
现在要对虚拟障碍做一些调整，具体来说，要根据台阶的深度即step_width来设置虚拟圆柱向外的偏移距离，当台阶深度=30cm时，保持半径5cm，偏移为0；当台阶深度小于30cm后，例如台阶深度为25cm，需要先减去机器人脚长20cm，剩余5cm安全余量，则凸边缘内侧与凹边缘外侧虚拟障碍距离分别为5cm/2=2.5cm，此时对于凸边缘处来说，由于外侧5cm，内侧2.5cm，因此轴线应该向外偏移1.25cm，而凹边缘直接向内偏移2.5cm即可。