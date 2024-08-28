# 7.28 add relu 展示CINN中的IR 
# 
# FLAGS_enable_pir_api=1 GLOG_v=6 python add_relu.py > add_relu_GLOGv6.txt
import unittest
import numpy as np
import paddle
from paddle.static import InputSpec

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x + y
        out = paddle.nn.functional.relu(z)
        return out
    
# Step 1: 构建模型对象，并应用动转静策略
specs = [InputSpec(shape=(-1, -1)), InputSpec(shape=(-1, -1))]
net = paddle.jit.to_static(SimpleNet(), specs)

# Step 2: 准备输入，执行 forward
x = paddle.rand(shape=[16, 64], dtype=paddle.float32)
y = paddle.rand(shape=[16, 64], dtype=paddle.float32)

print("device name: ", paddle.device.cuda.get_device_name())


out = net(x, y)

# print(out)
