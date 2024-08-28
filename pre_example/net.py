# 
# 上述代码示例中我们创建了一个简单的 rms_norm 计算子图
# 使用飞桨的动转静流程将子图转为静态图并调用编译器 CINN 进行优化和执行 
# 

import paddle
import timeit
from paddle import nn
from paddle.static import InputSpec

# 定义神经网络
class RMSNorm(nn.Layer):
    def __init__(self):
        super().__init__()
        paddle.seed(2024)
        self.hidden_size = 768
        self.weight = paddle.randn([self.hidden_size], dtype="float32")
        self.variance_epsilon = 1e-6

    def forward(self, hidden_states):
        variance = (hidden_states * hidden_states).sum(-1, keepdim=True) / 768
        hidden_states = (
            paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
        )
        return hidden_states * self.weight


def run_net(input_data):
    net = RMSNorm()

    # 指定输入变量的维度、数据类型等信息，具体接口可参考：
    # https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/jit/basic_usage_cn.html#inputspec
    input_spec = [
        InputSpec(shape=[1, None, 768], dtype='float32'),
    ]
    net = paddle.jit.to_static(
            net,
            input_spec=input_spec,
            full_graph=True,
        )
    # 使用 eval 模式
    net.eval()
    # 执行计算图
    out = net(input_data)
    return out

# 创建输入数据
input_data = paddle.randn([1, 2048, 768], dtype="float32")
print("device name: ", paddle.device.cuda.get_device_name())


warmup = 10
repeat_times = 100

for _ in range(warmup + repeat_times):
    if _ == warmup-1:
        start_time = timeit.default_timer()
    out = run_net(input_data)

# start_time = timeit.default_timer()
# out = run_net(input_data)

paddle.device.synchronize()
end_time = timeit.default_timer()

print("Time costs: {:.3f} ms / time".format((end_time - start_time)* 1000 / repeat_times))

# print(out)
