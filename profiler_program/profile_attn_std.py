# 9.02 Profile attention_standard 
# 
# 使用Paddle Profiler来分析程序性能 实验性使用
#
# python profile_attn_std.py
# python profile_attn_std.py -batch_size 16 -head_dim 32 -seq_len 1024
import timeit
import argparse 
import config
import paddle
from paddle.static import InputSpec
import paddle.profiler as profiler
from utils import print_hyperparameter, print_gpu_specific

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
    
    query = paddle.rand([batch_size, num_heads, seq_len, head_dim], dtype=paddle.float16)  
    key = paddle.rand([batch_size, num_heads, seq_len, head_dim], dtype=paddle.float16)  
    value = paddle.rand([batch_size, num_heads, seq_len, head_dim], dtype=paddle.float16)  
    query.stop_gradient = False
    key.stop_gradient = False
    value.stop_gradient = False
   
    input_spec = [
        InputSpec(shape=[batch_size, num_heads, seq_len, head_dim], dtype='float16'),
        InputSpec(shape=[batch_size, num_heads, seq_len, head_dim], dtype='float16'),
        InputSpec(shape=[batch_size, num_heads, seq_len, head_dim], dtype='float16'),
    ]

    prof = profiler.Profiler(targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
                   scheduler = (20, 120),
                   on_trace_ready = profiler.export_chrome_tracing('./profiler_log'),
                   timer_only = False)

    prof.start()
    for _ in range(warmup_times + running_times):
        if _ == warmup_times-1:
            start_time = timeit.default_timer()
        out = attention(query, key, value)
        prof.step()
    prof.stop()
        
    paddle.device.synchronize()
    end_time = timeit.default_timer()
    
    prof.summary(sorted_by=profiler.SortedKeys.GPUTotal,
            op_detail=True,
            thread_sep=False,
            time_unit='ms')

    print("StandAttention | {:2d} | {:2d} | {:4d} |  Time costs: {:.3f} ms / time".format(batch_size, head_dim, seq_len, (end_time - start_time) * 1000 / running_times)) 