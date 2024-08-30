# Songguo-paddle 
+ `/pre_example` 是 7月底做的尝试，实现了初步的attention测试
+ `/output` 记录了部分信息输出； `/log` 记录了 程序的运行结果
+ 相比于之前的代码，将所有超参数设置放进`config.py`；常用信息输出在 `utils.py` 

## 对比测试主体
+ 初期规模测试的三种方案：
    1. `attention_standard.py` 用函数的形式实现，最标准的attention；就是天王老子来了这个也是对的
    2. `attention_flash.py` 调用paddle-api中的`flash_attn`，paddle文档没说，但源码有；flashattention主页也可点进去
    3. `attention_cinn.py` 需要配合 run_attn_cinn.sh 来使，涉及到了动转静，class包起来的Attention，逻辑同standard


## 备注
+ 主体代码和docker环境均部署在A100服务器上，一切测试结果以A100为准（sm80｜40GB）
+ paddle-api中的 `flash_attn` 和我的实现在维度上有一定区别，我的实现是标准的