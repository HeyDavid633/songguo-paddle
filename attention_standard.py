# 8.28 attention_standard 
# 
# 只测试fp16精度 因为 attention_flash只支持fp16
# 此处为自己手写的attention标准计算流程，对于class Attnetion包起来的部分 对性能影响不大
#
# python attention_standard.py
# python attention_standard.py -batch_size 16 -head_dim 32 -seq_len 128  
import timeit
import argparse 
import config
import paddle
from paddle.static import InputSpec
from utils import print_hyperparameter, print_gpu_specific

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type = int, required=False, help='batch_size which 1 | 8 | 16')
    parser.add_argument('-head_dim', type = int, required=False, help='head_dim which 32 | 64')
    parser.add_argument('-seq_len', type = int, required=False, help='sequence length 128 ~ 2k, FlashAtten max as 16k in H100(80G)')
    args = parser.parse_args()
    
    if args.batch_size != None:
        batch_size = args.batch_size
    else:
        batch_size = config.BATCH_SIZE
    if args.head_dim != None:
        head_dim = args.head_dim
    else:
        head_dim = config.HEAD_DIM
    if args.seq_len != None:
        seq_len = args.seq_len
    else:
        seq_len = config.SEQ_LEN
    
    num_heads = config.HEADS_NUM
    warmup_times = config.WARMUP_TIME
    running_times = config.RUNNING_TIME
    
    # print_hyperparameter(batch_size, seq_len, num_heads, head_dim, warmup_times, running_times)
    # print_gpu_specific()
    
    
    # randn for init input
    paddle.seed(1)
    
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
    
    input_spec = [
        InputSpec(shape=[batch_size, num_heads, seq_len, head_dim], dtype='float16'),
        InputSpec(shape=[batch_size, num_heads, seq_len, head_dim], dtype='float16'),
        InputSpec(shape=[batch_size, num_heads, seq_len, head_dim], dtype='float16'),
    ]


    net = Attention()
    net = paddle.jit.to_static(
        net,
        input_spec=input_spec,
        full_graph=True,
    )
    net.eval()

    for _ in range(warmup_times + running_times):
        if _ == warmup_times-1:
            start_time = timeit.default_timer()
        out = attention(query, key, value)
        
    paddle.device.synchronize()
    end_time = timeit.default_timer()

    print("StandAttention | {:2d} | {:2d} | {:4d} |  Time costs: {:.3f} ms / time".format(batch_size, head_dim, seq_len, (end_time - start_time) * 1000 / running_times)) 
        
    attn_standard_filename = 'attn_standard_time.txt'
    with open(attn_standard_filename, 'a') as f: 
        print("StandAttention | {:2d} | {:2d} | {:4d} |  Time costs: {:.3f} ms / time".format(batch_size, head_dim, seq_len, (end_time - start_time) * 1000 / running_times), file=f) 
