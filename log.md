## 20260603
- 重新clone仓库
- 修改参parkour配置中的参考运动路径
- 用的isaaclab_510环境，缺失pkl.py，从IsaacLab源码中复制到该环境的包目录下
- 用onnx模型play时回报错，修改play.py，做如下修改：
    - 正常找 .pt 优先：先用 get_checkpoint_path 找，找到就正常加载 PyTorch checkpoint
    - 纯 ONNX 回退：如果是 --useonnx 且找不到 .pt，才回退到 resume_path = None
    - 对比逻辑 guard：只有当确实加载了 PyTorch checkpoint 时，才做 ONNX vs PyTorch 对比
- 新增ROUGH_TERRAINS_CFG_PLAY，避免每次修改训练时地形配置，同时修改对应的速度指令配置为仅包含对应地形
- play模式下相机视角改为world(便于鼠标拖拽观察)
- 将Q映射为play模式下的减速键，步长都从0.5改为0.1
- 金字塔地形添加step_width_range配置,台阶深度按0.01步长采样
------------------------------------------------------------
以下修改 从**20260514_225303**resume训练5000次迭代后，机器人跳着走
- 将速度指令改为范围内随机采样，线速度按0.1步长采样
- 关闭heading_cmd，因为它不是我理解的那种航向控制（可能也还需要确认）
- 地形开启use_cache
- onnx和pytorch模型输出没有差异，不要甩锅给这个
------------------------------------------------------------
- **报错：** `error: 无法推送一些引用到 'https://github.com/ZZQ0316/InstinctLab.git'`
后面 git lfs ls-files -s 排查发现是commit中包含下面这个文件
source/instinctlab/instinctlab/assets/resources/unitree_g1/g1.png
有可能是它偏大，但其实也只有412kb
只能把它从commit中删除,然后再push
`git rm --cached source/instinctlab/instinctlab/assets/resources/unitree_g1/g1.png`
`git commit --amend`
`git push`

## 20260604
- 地形配置 平地0.05 stand0.1 ps0.4 ps_inv0.35 slope0.1
- 下楼速度范围 从[0.45,0.8]降低为[0.3,0.6]
- 用不带鞋的urdf
- 训练30000次迭代，每2000次保存一下checkpoint
- 虚拟障碍修改
    删除内凹边缘的虚拟障碍
    外凸边缘的障碍向外、向上分别移动5cm
- step_width改为从{0.21, 0.27, 0.3}中随机采样
- 楼梯地形噪声系数降低为0.03
- 5cm台阶垂直修正 slope_threshold改为0.8
- gpu_collision_stack_size = 2**30, 改大了
---------------------------------------------------------
- 得到模型 20260605_110544
- 实测·效果没有之前的好，但差别也不大

## 20260612
- 从 https://github.com/Roboparty/robolab 移植过来一些camera noise pipeline

## 20260615
- 从 **20260605_110544** resume训练
- 地形配置 平地0.05 stand0.1 ps0.3 ps_inv0.3 slope0.05 boxes0.2
- 下楼速度范围改回0.45,0.8
- volume惩罚扩大到-5，两个速度奖励降低为1.5
- 虚拟障碍半径保持0.05
    向上偏移改为0.4，水平向外偏移改为0.06
- 去掉楼梯和boxes地形中所有的噪声
- 每500次迭代保存一次
- 要不要训练一个MLP，用于自主选择最佳速度？
    - 暂时不要，没有想象中那么简单
---------------------------------------------------------
- 还是无法适应小尺寸地形
- 但加入boxes地形后，上楼时最后一级台阶不会再像之前一样不抬腿：经验证就是之前没加box的原因
- 但是微调前的小楼梯行走能力也不在了


## 20260623
- 虚拟障碍改回上、外各0.05偏移
- 地形配置同上
    小尺寸地形比例多一些
- 奖励权重恢复原来，即volume惩罚4.0，速度奖励各2.0
    volume惩罚权重已经较大，没必要再降低速度追踪奖励权重
- heading_error惩罚加大 -1.0变为-1.5
-----------------------------------------------------------
训练了4000次迭代，还是和上次一样，安全落脚能力丢了
原因分析：可能是resume训练导致落脚能力无法完全恢复，但说明不是修改了虚拟障碍偏移的问题，也不是奖励权重修改的原因

## 20260624
- heading_error改回-1.0
- 地形配置见本地文档
- 虚拟障碍优化（见note.md）
--------------------------------------------------
- 上楼大部分时是正常的，下楼依旧很差，且比之前还要差；
- 速度感觉跟踪的并不好，虽然reward曲线正常；
- 平地走路也轻微抬腿了，不合理；

## 20260710微调
- 地形配置同上次
- 虚拟障碍：
    内边缘固定障碍 半径2cm上移2cm
    外边缘固定障碍 半径2.5cm外移2.5cm
- 奖励配置
    支撑面积 ok 哪只脚触地就对哪只脚计算奖励
    前缘净空 no
    落足稳定 no 
----------------------------------------------------
在上次基础上微调，结果机器人步态很差，表现为右脚被拖着前进；
下楼效果也并没有改善；
feet_slide惩罚在微调过程中变大
新加的feet_contact_area奖励尺寸太大（max1.0*2*weight2.0=4）

## 20260715从头训练
- 原则
    脚可以越过边缘一点，只要仍然能够稳定支撑身体。
    真正决定稳定性的是还有多少脚底在支撑
    要惩罚脚尖撞竖面，以免绊倒
- 足部接触点向脚尖及脚跟扩展
    x_min: -0.025 -> -0.04
    x_max: 0.12 -> 0.14
    x_num: 10 -> 12
- 边缘穿透惩罚
    凹边缘圆柱半径2.5cm，向上偏移2.5cm
    凸边缘内侧取消，外侧保留半径5cm，共1/2个圆柱，向外偏移2cm
    似乎可以根据地形step_width参数自适应设置，但这次先不改了吧
- 脚踢立面惩罚
    暂无
- feet_contact_area权重变为1.0
- 地形变化
- 速度变化
    下楼速度范围适当缩小变为[0.3,0.6]
- 深度相机的noise_pipeline还是需要丰富
    似乎不加的话，sim2real会非常差
    添加camera offset随机化
    添加rangebasedgaussiannoise










## Equivariant Latent-Space Symmetry Augmentation
这是一个在潜在特征空间中进行镜像增强的方法，用于让人形机器人学习到左右对称的行为。核心思想是：不是在高维输入空间做数据增强，而是在编码后的低维潜在空间做镜像变换。
# Step 1: 构建等变编码器
编码器必须保证：镜像输入 → 镜像潜在向量

1.1 等变线性层
class EquivariantLinear(nn.Module):
    """
    等变线性层：f(M_c(x)) = M_c(f(x))
    权重矩阵为块对称结构：[[A, B], [B, A]]，偏置共享
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 输入输出通道数都是偶数（左右各半）
        assert in_channels % 2 == 0 and out_channels % 2 == 0
        self.in_c = in_channels // 2
        self.out_c = out_channels // 2
        
        self.A = nn.Parameter(torch.randn(self.out_c, self.in_c))
        self.B = nn.Parameter(torch.randn(self.out_c, self.in_c))
        self.bias = nn.Parameter(torch.zeros(self.out_c))
    
    def forward(self, x):
        # x shape: (batch, 2 * in_c)
        left, right = x.chunk(2, dim=-1)
        out_left = left @ self.A.T + right @ self.B.T + self.bias
        out_right = left @ self.B.T + right @ self.A.T + self.bias
        return torch.cat([out_left, out_right], dim=-1)

1.2 等变 MLP
class EquivariantMLP(nn.Module):
    """由等变线性层堆叠而成"""
    def __init__(self, in_dim, out_dim, hidden_dims=[512, 256, 128]):
        super().__init__()
        dims = [in_dim] + hidden_dims + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(EquivariantLinear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ELU())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

1.3 等变 CNN（处理深度图像）
class LiftConv(nn.Module):
    """提升卷积：将普通图像提升为对称通道表示"""
    def __init__(self, in_ch=1, out_ch=32, kernel=8, stride=4):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride)
        self.bias = nn.Parameter(torch.zeros(out_ch))
    
    def forward(self, x):
        # x: (batch, 1, H, W)
        f = self.conv(x)  # (batch, C, H', W')
        f_mirror = torch.flip(f, dims=[-1])  # 水平翻转
        # 输出两个通道组
        return torch.stack([f + self.bias, f_mirror + self.bias], dim=1)

class EquivariantConv(nn.Module):
    """等变卷积：保持左右通道对称"""
    def __init__(self, in_c, out_c, kernel=3, stride=2, padding=1):
        super().__init__()
        self.conv_a = nn.Conv2d(in_c, out_c, kernel, stride, padding)
        self.conv_b = nn.Conv2d(in_c, out_c, kernel, stride, padding)
    
    def forward(self, x):
        # x: (batch, 2, C, H, W)
        left, right = x[:, 0], x[:, 1]
        out_left = self.conv_a(left) + self.conv_b(right)
        out_right = self.conv_b(left) + self.conv_a(right)
        return torch.stack([out_left, out_right], dim=1)

# Step 2: 输入结构化
def structure_proprioception(proprio_history):
    """
    将本体感知历史重组为左右对称结构
    输入: (batch, h, 72) — h帧历史
    输出: (batch, 84) — 左通道42维 + 右通道42维
    """
    # 根据关节对应关系，将左腿/右腿、左臂/右臂分离
    left_features = extract_left_features(proprio_history)   # (batch, 42)
    right_features = extract_right_features(proprio_history) # (batch, 42)
    return torch.cat([left_features, right_features], dim=-1)

# Step 3: 潜在空间镜像操作
def mirror_latent(z):
    """
    在潜在空间做镜像：直接交换左右通道组
    输入: (batch, 2 * C)
    输出: (batch, 2 * C) — 左右通道互换
    """
    left, right = z.chunk(2, dim=-1)
    return torch.cat([right, left], dim=-1)

def mirror_action(action, joint_mapping):
    """
    镜像动作：交换左右关节并调整符号
    输入: (batch, 21) — 关节目标位置
    """
    mirrored = action.clone()
    for left_idx, right_idx, sign in joint_mapping:
        mirrored[:, left_idx] = action[:, right_idx] * sign
        mirrored[:, right_idx] = action[:, left_idx] * sign
    return mirrored

# Step 4: 训练时应用增强
def train_step(encoder, actor, critic, batch, joint_mapping):
    depth, proprio, action, adv = batch
    
    # 1. 原始样本的前向
    structured_proprio = structure_proprioception(proprio)
    latent = encoder(depth, structured_proprio)  # (batch, 48)
    
    # 2. 生成镜像样本（只在潜在空间操作）
    latent_mirror = mirror_latent(latent)
    action_mirror = mirror_action(action, joint_mapping)
    
    # 3. 计算 loss（原始 + 镜像各一半）
    value = critic(obs, latent)
    value_mirror = critic(obs_mirror, latent_mirror)
    
    loss = 0.5 * ppo_loss(value, action, adv) + \
           0.5 * ppo_loss(value_mirror, action_mirror, adv)
    
    return loss