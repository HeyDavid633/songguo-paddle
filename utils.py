import paddle  
  
def check_equal(tensor_a, tensor_b, tolerance=0.0001):  
    """  
    比较两个张量的元素是否相等，考虑到精度误差。  
      
    :param tensor_a: 第一个张量  
    :param tensor_b: 第二个张量  
    :param tolerance: 允许的最大误差，默认为0.001  
    :return: 相等比率（百分比）  
    """  
    # 确保两个张量的形状相同  
    assert tensor_a.shape == tensor_b.shape, "The shapes of the two tensors must be the same."  
  
    # 创建一个与 diff 形状相同的张量，其所有元素都等于 tolerance  
    tolerance_tensor = paddle.full(shape=tensor_a.shape, fill_value=tolerance, dtype=tensor_a.dtype)  
  
    # 计算两个张量之间的绝对差值  
    diff = paddle.abs(tensor_a - tensor_b)  
  
    # 检查哪些元素的差值在允许的误差范围内  
    equal_elements = paddle.less_equal(diff, tolerance_tensor)  
  
    # 计算相等元素的数量  
    num_equal_elements = paddle.sum(paddle.cast(equal_elements, dtype='int32'))  
  
    # 计算总元素数量  
    total_elements = paddle.numel(tensor_a)  
  
    # 计算相等比率  
    equality_ratio = (num_equal_elements / total_elements) * 100  
  
    # 提取张量中的数值  
    equality_ratio_value = equality_ratio.item()  
  
    # 打印相等比率  
    print(f"Equality ratio: {equality_ratio_value:.2f}%")  