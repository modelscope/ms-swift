import os


def try_use_single_device_mode():
    if os.environ.get('SWIFT_SINGLE_DEVICE_MODE', '0') == '1':
        visible_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        visible_device = visible_devices[int(os.environ['LOCAL_RANK'])]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(visible_device)
        os.environ['LOCAL_RANK'] = '0'
