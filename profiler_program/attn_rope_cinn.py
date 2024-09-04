# 9.03 attn_rope_cinn
#
# python attn_rope_cinn.py
# python attn_rope_cinn.py -batch_size 16 -head_dim 32 -seq_len 128  
import timeit
import argparse 
import config
import paddle
from paddle.static import InputSpec
from utils import print_hyperparameter, print_gpu_specific


def sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim):
    # (max_len, 1)
    position = paddle.arange(0, max_len, dtype=paddle.float16).unsqueeze(-1)
    # (output_dim//2)
    ids = paddle.arange(0, output_dim // 2, dtype=paddle.float16)  # 即公式里的i, i的范围是 [0,d/2]
    
    # print(ids)
    # print(-2 * ids / output_dim)
    
    tensor10000 = paddle.full(shape=[output_dim // 2], fill_value=10000.0, dtype=paddle.float16)   # dwh: 应对paddle做了修改 
    theta = paddle.pow(tensor10000, -2 * ids / output_dim)
    # (max_len, output_dim//2)
    embeddings = position * theta  # 即公式里的：pos / (10000^(2i/d))
    # (max_len, output_dim//2, 2)
    embeddings = paddle.stack([paddle.sin(embeddings), paddle.cos(embeddings)], axis=-1)
    # (bs, head, max_len, output_dim//2, 2)
    # 在bs维度重复，其他维度都是1不重复
    embeddings = paddle.tile(embeddings, repeat_times=[batch_size, nums_head] + [1] * len(embeddings.shape))  # dwh: 应对paddle做了修改 
    # (bs, head, max_len, output_dim)
    # reshape后就是：偶数sin, 奇数cos了
    embeddings = paddle.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))  
    # embeddings = embeddings.to(device)
    
    # print(embeddings)
    return embeddings

def RoPE(q, k):
    # q,k: (bs, head, max_len, output_dim)
    batch_size = q.shape[0]
    nums_head = q.shape[1]
    max_len = q.shape[2]
    output_dim = q.shape[-1]
    # (bs, head, max_len, output_dim)
    pos_emb = sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim)
    
    pos_emb = paddle.cast(pos_emb, dtype=paddle.float32)  # repeat_interleave不支持float16的精度
    
    # cos_pos,sin_pos: (bs, head, max_len, output_dim)  看rope公式可知，相邻cos，sin之间是相同的，所以复制一遍。如(1,2,3)变成(1,1,2,2,3,3)
    cos_pos = pos_emb[...,  1::2].repeat_interleave(2, axis=-1)  # 将奇数列信息抽取出来也就是cos 拿出来并复制
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, axis=-1)  # 将偶数列信息抽取出来也就是sin 拿出来并复制
    
    cos_pos = paddle.cast(cos_pos, dtype=paddle.float16)
    sin_pos = paddle.cast(sin_pos, dtype=paddle.float16)
    
    # q,k: (bs, head, max_len, output_dim)
    q2 = paddle.stack([-q[..., 1::2], q[..., ::2]], axis=-1)
    q2 = q2.reshape(q.shape)  # reshape后就是正负交替了
    # 更新qw, *对应位置相乘
    q = q * cos_pos + q2 * sin_pos

    k2 = paddle.stack([-k[..., 1::2], k[..., ::2]], axis=-1)
    k2 = k2.reshape(k.shape)
    # 更新kw, *对应位置相乘
    k = k * cos_pos + k2 * sin_pos
    return q, k

class Attention(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, use_RoPE=True):
        head_dim = query.shape[-1]
        if use_RoPE:
            query, key = RoPE(query, key) 
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
    print_gpu_specific()

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
        out = net(query, key, value, use_RoPE=True)
        
        
    paddle.device.synchronize()
    end_time = timeit.default_timer()

    print("RoPE CINN Attn | {:2d} | {:2d} | {:4d} |  Time costs: {:.3f} ms / time".format(batch_size, head_dim, seq_len, (end_time - start_time) * 1000 / running_times)) 
        
    attn_cinn_filename = 'attn_rope_cinn_time.txt'
    with open(attn_cinn_filename, 'a') as f: 
        print("RoPE CINN Attn | {:2d} | {:2d} | {:4d} |  Time costs: {:.3f} ms / time".format(batch_size, head_dim, seq_len, (end_time - start_time) * 1000 / running_times), file=f) 
