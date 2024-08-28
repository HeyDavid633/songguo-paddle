import paddle

paddle.device.set_device('gpu')

print("paddle cudnn version: ", paddle.version.cudnn())

print("device name: ", paddle.device.cuda.get_device_name())
