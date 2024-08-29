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
# python attention_flash.py -batch_size 16 -seq_len 128 

import timeit
import argparse 
import config
import paddle
import utils

def attention(query, key, value):
    head_dim = query.shape[-1]
    scores = paddle.matmul(query, key, transpose_y=True) / (head_dim ** 0.5) 
    probs = paddle.nn.functional.softmax(scores, axis=-1) 
    h = paddle.matmul(probs, value) 
    return h

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type = int, required=False, help='batch_size which 1 | 8 | 16')
    parser.add_argument('-seq_len', type = int, required=False, help='sequence length 128 ~ 2k, FlashAtten max as 16k in H100(80G)')
    args = parser.parse_args()
    
    if args.batch_size != None:
        batch_size = args.batch_size
    else:
        batch_size = config.BATCH_SIZE
    if args.seq_len != None:
        seq_len = args.seq_len
    else:
        seq_len = config.SEQ_LEN
    
    num_heads = config.HEADS_NUM
    head_dim = config.HEAD_DIM
    warmup_times = config.WARMUP_TIME
    running_times = config.RUNNING_TIME
    
    print("-"*5, "Hyperparameter Argument", "-"*5)
    print("{:13} : {:5}".format('batch_size', batch_size))
    print("{:13} : {:5}".format('seq_len', seq_len))
    print("{:13} : {:5}".format('num_heads', num_heads))
    print("{:13} : {:5}\n".format('head_dim', head_dim))
    print("{:13} : {:5}".format('warmup_times', warmup_times))
    print("{:13} : {:5}".format('running_times', running_times))
    print("-"*30)
    
    # set running device as gpu | specfic --> A100
    paddle.device.set_device('gpu')
    print("device name: ", paddle.device.cuda.get_device_name())
    
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
    
    output_our = attention(query, key, value)
    tuple_output_flash = paddle.nn.functional.flash_attention.flash_attention(query1, key1, value1)
    output_flash = tuple_output_flash[0]
    output_flash = paddle.transpose(output_flash, [0, 2, 1, 3])
        
    
    print("output_flash:  ", output_flash)
    print("output_our:    ", output_our)
    
    utils.check_equal(output_flash, output_flash)
    
    # print(paddle.equal(output_flash, output_our))
    