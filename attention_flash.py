# 8.29 flash_attention 
# 
# 只测试fp16的精度，与attention_standard对齐
# flash_attention的维度信息和我们的有一定差异，需要进行转换；当前版本已修改
#
# 直接调用 paddle内集成的flash-attention 
# paddle.nn.functional.flash_attention.flash_attention_with_sparse_mask() 必须使用稀疏mask ，文档中有写
# paddle.nn.functional.flash_attention.flash_attention() 文档中没提 --- 源代码有写，可以使用
# 
# 
# python attention_flash.py
# python attention_flash.py -batch_size 16 -head_dim 32 -seq_len 128 

import timeit
import argparse 
import config
import paddle
from utils import print_hyperparameter, check_equal, print_gpu_specific

def attention(query, key, value):
    head_dim = query.shape[-1]
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
    
    # query1 = paddle.rand([batch_size, seq_len, num_heads, head_dim], dtype=paddle.float16)  
    # key1 = paddle.rand([batch_size, seq_len, num_heads, head_dim], dtype=paddle.float16)  
    # value1 = paddle.rand([batch_size, seq_len, num_heads, head_dim], dtype=paddle.float16)  
    query = paddle.rand([batch_size, num_heads, seq_len, head_dim], dtype=paddle.float16)  
    key = paddle.rand([batch_size, num_heads, seq_len, head_dim], dtype=paddle.float16)  
    value = paddle.rand([batch_size, num_heads, seq_len, head_dim], dtype=paddle.float16)  
    query1 = paddle.transpose(query, [0, 2, 1, 3])
    key1 = paddle.transpose(key, [0, 2, 1, 3])
    value1 = paddle.transpose(value, [0, 2, 1, 3])
    
    # to check
    output_our = attention(query, key, value)
    
    for _ in range(warmup_times + running_times):
        if _ == warmup_times-1:
            start_time = timeit.default_timer()
        tuple_output_flash = paddle.nn.functional.flash_attention.flash_attention(query1, key1, value1)
        output_flash = tuple_output_flash[0]
        output_flash = paddle.transpose(output_flash, [0, 2, 1, 3])
    
    paddle.device.synchronize()
    end_time = timeit.default_timer()
        
    check_equal(output_flash, output_flash)
    
    print("FlashAttention | {:2d} | {:2d} | {:4d} |  Time costs: {:.3f} ms / time".format(batch_size, head_dim, seq_len, (end_time - start_time) * 1000 / running_times)) 
    
    attn_flash_filename = 'attn_flash_time.txt'
    with open(attn_flash_filename, 'a') as f: 
        print("FlashAttention | {:2d} | {:2d} | {:4d} |  Time costs: {:.3f} ms / time".format(batch_size, head_dim, seq_len, (end_time - start_time) * 1000 / running_times), file=f) 

    