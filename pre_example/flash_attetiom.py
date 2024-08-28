# 7.27 paddle flash_attention 
# 
# 这个接口只支持 float16、bfloat16 --- PyTorch中的flash_attention只支持fp16
# 
# 直接调用 paddle内集成的flash-attention 
# paddle.nn.functional.flash_attention.flash_attention_with_sparse_mask() 必须使用稀疏mask ，文档中有写
# paddle.nn.functional.flash_attention.flash_attention() 文档中没提 --- 源代码有写，可以使用
#
import timeit
import paddle
import numpy as np
paddle.device.set_device('gpu')

# 超参数定义
batch_size = 16
num_heads = 16
seq_len = 128
head_dim = 32

warmup = 100
repeat_times = 100

print("device name: ", paddle.device.cuda.get_device_name())

query = paddle.rand([batch_size, num_heads, seq_len, head_dim], dtype=paddle.float16)  
key = paddle.rand([batch_size, num_heads, seq_len, head_dim], dtype=paddle.float16)  
value = paddle.rand([batch_size, num_heads, seq_len, head_dim], dtype=paddle.float16)  
query.stop_gradient = False
key.stop_gradient = False
value.stop_gradient = False

# mask = paddle.nn.functional.dropout(paddle.ones([seq_len, seq_len])).expand([batch_size, num_heads, seq_len, seq_len])
# sp_mask = mask.reshape([-1, seq_len, seq_len]).to_sparse_csr()
# kp_mask = paddle.randint(0, 2, [batch_size, seq_len]).astype(paddle.float16)
# attn_mask = paddle.randint(0, 2, [seq_len, seq_len]).astype(paddle.float16)


for _ in range(warmup + repeat_times):
    if _ == warmup-1:
        start_time = timeit.default_timer() 
    output = paddle.nn.functional.flash_attention.flash_attention(query, key, value)
# output.backward()

paddle.device.synchronize()
end_time = timeit.default_timer()

print(output)
      
print("Seqlen: {}, Time costs: {:.3f} ms / time".format(seq_len, (end_time - start_time) * 1000 / repeat_times)) 