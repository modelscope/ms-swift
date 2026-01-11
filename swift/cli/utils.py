import os
import argparse

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
        
def try_get_proc_title():
    try:
        from setproctitle import setproctitle
    except ImportError:
        return
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--proc_title', type=str, default=os.environ.get('SWIFT_PROC_TITLE', 'ms-swift'))
    args, _ = parser.parse_known_args()
    setproctitle(args.proc_title)