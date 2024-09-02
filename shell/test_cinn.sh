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

python attention_standard_cinn.py -batch_size 1 -head_dim 32 -seq_len 128
python attention_standard_cinn.py -batch_size 1 -head_dim 32 -seq_len 256
python attention_standard_cinn.py -batch_size 1 -head_dim 32 -seq_len 512
python attention_standard_cinn.py -batch_size 1 -head_dim 32 -seq_len 1024
python attention_standard_cinn.py -batch_size 1 -head_dim 32 -seq_len 2048
python attention_standard_cinn.py -batch_size 1 -head_dim 32 -seq_len 4096
python attention_standard_cinn.py -batch_size 8 -head_dim 32 -seq_len 128
python attention_standard_cinn.py -batch_size 8 -head_dim 32 -seq_len 256
python attention_standard_cinn.py -batch_size 8 -head_dim 32 -seq_len 512
python attention_standard_cinn.py -batch_size 8 -head_dim 32 -seq_len 1024
python attention_standard_cinn.py -batch_size 8 -head_dim 32 -seq_len 2048
python attention_standard_cinn.py -batch_size 8 -head_dim 32 -seq_len 4096
python attention_standard_cinn.py -batch_size 16 -head_dim 32 -seq_len 128
python attention_standard_cinn.py -batch_size 16 -head_dim 32 -seq_len 256
python attention_standard_cinn.py -batch_size 16 -head_dim 32 -seq_len 512
python attention_standard_cinn.py -batch_size 16 -head_dim 32 -seq_len 1024
python attention_standard_cinn.py -batch_size 16 -head_dim 32 -seq_len 2048
python attention_standard_cinn.py -batch_size 16 -head_dim 32 -seq_len 4096
python attention_standard_cinn.py -batch_size 1 -head_dim 64 -seq_len 128
python attention_standard_cinn.py -batch_size 1 -head_dim 64 -seq_len 256
python attention_standard_cinn.py -batch_size 1 -head_dim 64 -seq_len 512
python attention_standard_cinn.py -batch_size 1 -head_dim 64 -seq_len 1024
python attention_standard_cinn.py -batch_size 1 -head_dim 64 -seq_len 2048
python attention_standard_cinn.py -batch_size 1 -head_dim 64 -seq_len 4096
python attention_standard_cinn.py -batch_size 8 -head_dim 64 -seq_len 128
python attention_standard_cinn.py -batch_size 8 -head_dim 64 -seq_len 256
python attention_standard_cinn.py -batch_size 8 -head_dim 64 -seq_len 512
python attention_standard_cinn.py -batch_size 8 -head_dim 64 -seq_len 1024
python attention_standard_cinn.py -batch_size 8 -head_dim 64 -seq_len 2048
python attention_standard_cinn.py -batch_size 8 -head_dim 64 -seq_len 4096
python attention_standard_cinn.py -batch_size 16 -head_dim 64 -seq_len 128
python attention_standard_cinn.py -batch_size 16 -head_dim 64 -seq_len 256
python attention_standard_cinn.py -batch_size 16 -head_dim 64 -seq_len 512
python attention_standard_cinn.py -batch_size 16 -head_dim 64 -seq_len 1024
python attention_standard_cinn.py -batch_size 16 -head_dim 64 -seq_len 2048
python attention_standard_cinn.py -batch_size 16 -head_dim 64 -seq_len 4096