import os


def try_use_single_device_mode():
    if os.environ.get('SWIFT_SINGLE_DEVICE_MODE', '0') == '1':
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        local_rank = os.environ.get('LOCAL_RANK')
        if local_rank is None or not visible_devices:
            return
        visible_devices = visible_devices.split(',')
        visible_device = visible_devices[int(local_rank)]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(visible_device)
        os.environ['LOCAL_RANK'] = '0'
