# 7.26 paddle attention 
# 
# 这个接口只支持 float32、float64
# 
# 直接调用paddle内置的paddle.sparse.nn.functional.attention---  没有去纠结它的实现
# 介绍：稀疏 Attention，该 API 内部使用 SparseCsrTensor （CSR）来存储 Transformer 模块中的 attention 矩阵
# 参数 sparse_mask 描述了稀疏矩阵的非 0 元素索引布局。
#
import timeit
import paddle
from paddle.static import InputSpec
paddle.device.set_device('gpu')

# 超参数定义
batch_size = 16
num_heads = 16
seq_len = 128
head_dim = 32

warmup = 100
repeat_times = 100

print("device name: ", paddle.device.cuda.get_device_name())

query = paddle.rand([batch_size, num_heads, seq_len, head_dim], dtype=paddle.float32)  
key = paddle.rand([batch_size, num_heads, seq_len, head_dim], dtype=paddle.float32)  
value = paddle.rand([batch_size, num_heads, seq_len, head_dim], dtype=paddle.float32)  
query.stop_gradient = False
key.stop_gradient = False
value.stop_gradient = False

mask = paddle.nn.functional.dropout(paddle.ones([seq_len, seq_len])).expand([batch_size, num_heads, seq_len, seq_len])
sp_mask = mask.reshape([-1, seq_len, seq_len]).to_sparse_csr()
kp_mask = paddle.randint(0, 2, [batch_size, seq_len]).astype(paddle.float32)
attn_mask = paddle.randint(0, 2, [seq_len, seq_len]).astype(paddle.float32)

class Attention(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        scores = paddle.matmul(query, key, transpose_y=True) / (head_dim ** 0.5) 
        probs = paddle.nn.functional.softmax(scores, axis=-1) 
        h = paddle.matmul(probs, value) 
        return h

def attention(query, key, value):
    scores = paddle.matmul(query, key, transpose_y=True) / (head_dim ** 0.5) 
    probs = paddle.nn.functional.softmax(scores, axis=-1) 
    h = paddle.matmul(probs, value) 
    return h


input_spec = [
        InputSpec(shape=[batch_size, num_heads, seq_len, head_dim], dtype='float32'),
        InputSpec(shape=[batch_size, num_heads, seq_len, head_dim], dtype='float32'),
        InputSpec(shape=[batch_size, num_heads, seq_len, head_dim], dtype='float32'),
    ]


net = Attention()
net = paddle.jit.to_static(
        net,
        input_spec=input_spec,
        full_graph=True,
    )
net.eval()

for _ in range(warmup + repeat_times):
    if _ == warmup-1:
        start_time = timeit.default_timer()
    out = net(query, key, value)
    
paddle.device.synchronize()
end_time = timeit.default_timer()

# print(h.shape)
      
print("Seqlen: {}, Time costs: {:.3f} ms / time".format(seq_len, (end_time - start_time) * 1000 / repeat_times)) 