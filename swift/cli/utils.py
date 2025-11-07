def is_ppu():
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)

    if result.returncode == 0:
        output = result.stdout
        return 'PPU-' in output
    else:
        return False


def fix_ppu():
    if is_ppu():
        import os
        visible_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        visible_device = visible_devices[int(os.environ['LOCAL_RANK'])]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(visible_device)
        os.environ['LOCAL_RANK'] = '0'
