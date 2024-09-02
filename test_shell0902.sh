# 测试脚本 test_0902.sh 
# 为松果项目规模化测试，程序均为最基本的Attention操作
# warmup 20 + running 100

batch_sizes=(1 8 16)
head_dims=(32 64)
seq_lens=(128 256 512 1024 2048 4096)
# batch_sizes=(1 8)
# head_dims=(32)
# seq_lens=(128)

for batch_size in "${batch_sizes[@]}"
do
    for head_dim in "${head_dims[@]}"
    do
        for seq_len in "${seq_lens[@]}"
        do
            # 执行命令
            python attention_flash.py -batch_size "$batch_size" -head_dim "$head_dim" -seq_len "$seq_len"
            python attention_standard.py -batch_size "$batch_size" -head_dim "$head_dim" -seq_len "$seq_len"
        done
    done
done

# 打开组合算子
export FLAGS_prim_enable_dynamic=true && export FLAGS_prim_all=true
# 打开 CINN 编译器相关 FLAG
export FLAGS_use_cinn=true
export FLAGS_cinn_new_group_scheduler=true
export FLAGS_group_schedule_tiling_first=true
export FLAGS_cinn_bucket_compile=true
# 打开 PIR 模式
export FLAGS_enable_pir_api=true
# 是否打印 Program IR 信息
export FLAGS_print_ir=false

for batch_size in "${batch_sizes[@]}"
do
    for head_dim in "${head_dims[@]}"
    do
        for seq_len in "${seq_lens[@]}"
        do
            python attention_standard_cinn.py -batch_size "$batch_size" -head_dim "$head_dim" -seq_len "$seq_len"
        done
    done
done