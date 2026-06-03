### 20260603
- 重新clone仓库
- 修改参parkour配置中的参考运动路径
- 用的isaaclab_510环境，缺失pkl.py，从IsaacLab源码中复制到该环境的包目录下
- 用onnx模型play时回报错，修改play.py，做如下修改：
1、正常找 .pt 优先：先用 get_checkpoint_path 找，找到就正常加载 PyTorch checkpoint
2、纯 ONNX 回退：如果是 --useonnx 且找不到 .pt，才回退到 resume_path = None
3、对比逻辑 guard：只有当确实加载了 PyTorch checkpoint 时，才做 ONNX vs PyTorch 对比
- 新增ROUGH_TERRAINS_CFG_PLAY，避免每次修改训练时地形配置，同时修改对应的速度指令配置为仅包含对应地形
- play模式下相机视角改为world(便于鼠标拖拽观察)
- 将Q映射为play模式下的减速键，步长都从0.5改为0.1
# 发现：
onnx和pytorch模型输出没有差异，不要甩锅给这个