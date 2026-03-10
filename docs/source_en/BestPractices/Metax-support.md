# Metax Support

## 1. use swift with Metax
you can either build an image or pull an existing one. Here, we demonstrate how to use ms-swift on Metax by pulling a pre-built image as an example.
### 1.1. start ms-swift Container
```bash
docker pull mx-devops-acr-cn-shanghai.cr.volces.com/opensource/public-ai-release/maca/ms-swift:3.10.3-maca.ai3.3.0.16-torch2.6-py310-ubuntu22.04-amd64
# you may modify privileged option and mount only specific GPU cards.
# please refer to our documents on https://developer.metax-tech.com
# Metax GPUs must be mounted via --device=/dev/dri --device=/dev/mxcd
docker run  -it --net=host --uts=host --ipc=host --privileged=true --group-add video  \
    --shm-size 100gb --ulimit memlock=-1 \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    --device=/dev/dri --device=/dev/mxcd \
    -v /root/workspace:/external \
    --name swift_test \
    mx-devops-acr-cn-shanghai.cr.volces.com/opensource/public-ai-release/maca/ms-swift:3.10.3-maca.ai3.3.0.16-torch2.6-py310-ubuntu22.04-amd64
```
## 2. Environment check
### 2.1. Check Metax available
Thanks to its compatibility with CUDA, we can use the same approach as NVIDIA to check the availability of Metax devices.
```python
import torch
print(torch.cuda.is_available())
# True
```
### 2.2. Check the P2P connections
```bash
mx-smi topo -m
# output
=================== MetaX System Management Interface Log ===================
Timestamp                                         : Wed Feb 11 16:37:10 2026

Attached GPUs                                     : 8
Device link type matrix
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    Node Affinity  CPU Affinity
GPU0    X       MX      MX      MX      NODE    NODE    NODE    NODE    0              0-31,64-95
GPU1    MX      X       MX      MX      NODE    NODE    NODE    NODE    0              0-31,64-95
GPU2    MX      MX      X       MX      NODE    NODE    NODE    NODE    0              0-31,64-95
GPU3    MX      MX      MX      X       NODE    NODE    NODE    NODE    0              0-31,64-95
GPU4    NODE    NODE    NODE    NODE    X       MX      MX      MX      0              0-31,64-95
GPU5    NODE    NODE    NODE    NODE    MX      X       MX      MX      0              0-31,64-95
GPU6    NODE    NODE    NODE    NODE    MX      MX      X       MX      0              0-31,64-95
GPU7    NODE    NODE    NODE    NODE    MX      MX      MX      X       0              0-31,64-95

Legend:
  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  MX   = Connection traversing MetaXLink
  ETH  = Connection traversing Eth
  NA   = Connection type is unknown
```
### 2.3. check the status of the GPUs
```bash
mx-smi
# output
    =================== MetaX System Management Interface Log ===================
Timestamp                                         : Wed Feb 11 09:55:49 2026

Attached GPUs                                     : 8
+---------------------------------------------------------------------------------+
| MX-SMI 2.2.9                       Kernel Mode Driver Version: 3.4.4            |
| MACA Version: 3.3.0.15             BIOS Version: 1.30.0.0                       |
|------------------+-----------------+---------------------+----------------------|
| Board       Name | GPU   Persist-M | Bus-id              | GPU-Util      sGPU-M |
| Pwr:Usage/Cap    | Temp       Perf | Memory-Usage        | GPU-State            |
|==================+=================+=====================+======================|
| 0     MetaX C500 | 0           Off | 0000:0e:00.0        | 0%          Disabled |
| 57W / 350W       | 35C          P0 | 826/65536 MiB       | Available            |
+------------------+-----------------+---------------------+----------------------+
| 1     MetaX C500 | 1           Off | 0000:0f:00.0        | 0%          Disabled |
| 58W / 350W       | 37C          P0 | 826/65536 MiB       | Available            |
+------------------+-----------------+---------------------+----------------------+
| 2     MetaX C500 | 2           Off | 0000:10:00.0        | 0%          Disabled |
| 58W / 350W       | 36C          P0 | 826/65536 MiB       | Available            |
+------------------+-----------------+---------------------+----------------------+
| 3     MetaX C500 | 3           Off | 0000:12:00.0        | 0%          Disabled |
| 60W / 350W       | 35C          P0 | 826/65536 MiB       | Available            |
+------------------+-----------------+---------------------+----------------------+
| 4     MetaX C500 | 4           Off | 0000:35:00.0        | 0%          Disabled |
| 57W / 350W       | 33C          P0 | 826/65536 MiB       | Available            |
+------------------+-----------------+---------------------+----------------------+
| 5     MetaX C500 | 5           Off | 0000:36:00.0        | 0%          Disabled |
| 56W / 350W       | 34C          P0 | 826/65536 MiB       | Available            |
+------------------+-----------------+---------------------+----------------------+
| 6     MetaX C500 | 6           Off | 0000:37:00.0        | 0%          Disabled |
| 55W / 350W       | 34C          P0 | 826/65536 MiB       | Available            |
+------------------+-----------------+---------------------+----------------------+
| 7     MetaX C500 | 7           Off | 0000:38:00.0        | 0%          Disabled |
| 56W / 350W       | 36C          P0 | 826/65536 MiB       | Available            |
+------------------+-----------------+---------------------+----------------------+

+---------------------------------------------------------------------------------+
| Process:                                                                        |
|  GPU                    PID         Process Name                 GPU Memory     |
|                                                                  Usage(MiB)     |
|=================================================================================|
|  no process found                                                               |
+---------------------------------------------------------------------------------+
```

## 3. run example
We support direct use of the community version. However, we also provide a more optimized version in the image under /workspace and strongly recommend using it.

### 3.1. run swift example
In most scenarios, we can run Swift's examples directly.
```bash
# We assume that the ms-swift code is under /workspace
cd /workspace/ms-swift/
bash examples/train/full/train.sh

```

```bash
# output:
{'loss': 1.47077751, 'grad_norm': 10.5625, 'learning_rate': 2e-06, 'token_acc': 0.65511727, 'epoch': 0.01, 'global_step/max_steps': '1/94', 'percentage': '1.06%', 'elapsed_time': '2s', 'remaining_time': '4m 28s', 'memory(GiB)': 4.87, 'train_speed(iter/s)': 0.345807}
{'loss': 1.58882141, 'grad_norm': 10.75, 'learning_rate': 1e-05, 'token_acc': 0.61763144, 'epoch': 0.05, 'global_step/max_steps': '5/94', 'percentage': '5.32%', 'elapsed_time': '10s', 'remaining_time': '3m 12s', 'memory(GiB)': 5.64, 'train_speed(iter/s)': 0.461462}
{'loss': 1.56617603, 'grad_norm': 12.8125, 'learning_rate': 9.92e-06, 'token_acc': 0.61519274, 'epoch': 0.11, 'global_step/max_steps': '10/94', 'percentage': '10.64%', 'elapsed_time': '20s', 'remaining_time': '2m 52s', 'memory(GiB)': 5.64, 'train_speed(iter/s)': 0.485796}
{'loss': 1.63347206, 'grad_norm': 13.6875, 'learning_rate': 9.69e-06, 'token_acc': 0.60373975, 'epoch': 0.16, 'global_step/max_steps': '15/94', 'percentage': '15.96%', 'elapsed_time': '30s', 'remaining_time': '2m 39s', 'memory(GiB)': 6.5, 'train_speed(iter/s)': 0.493855}
{'loss': 1.60613976, 'grad_norm': 11.0, 'learning_rate': 9.32e-06, 'token_acc': 0.59997221, 'epoch': 0.21, 'global_step/max_steps': '20/94', 'percentage': '21.28%', 'elapsed_time': '39s', 'remaining_time': '2m 27s', 'memory(GiB)': 6.5, 'train_speed(iter/s)': 0.500516}
{'loss': 1.45015478, 'grad_norm': 15.25, 'learning_rate': 8.8e-06, 'token_acc': 0.62373584, 'epoch': 0.27, 'global_step/max_steps': '25/94', 'percentage': '26.60%', 'elapsed_time': '49s', 'remaining_time': '2m 16s', 'memory(GiB)': 6.5, 'train_speed(iter/s)': 0.50548}
{'loss': 1.39427547, 'grad_norm': 13.9375, 'learning_rate': 8.18e-06, 'token_acc': 0.6357994, 'epoch': 0.32, 'global_step/max_steps': '30/94', 'percentage': '31.91%', 'elapsed_time': '59s', 'remaining_time': '2m 5s', 'memory(GiB)': 6.5, 'train_speed(iter/s)': 0.508409}
{'loss': 1.53672237, 'grad_norm': 11.125, 'learning_rate': 7.45e-06, 'token_acc': 0.61650612, 'epoch': 0.37, 'global_step/max_steps': '35/94', 'percentage': '37.23%', 'elapsed_time': '1m 8s', 'remaining_time': '1m 55s', 'memory(GiB)': 6.5, 'train_speed(iter/s)': 0.510425}
{'loss': 1.54039021, 'grad_norm': 13.8125, 'learning_rate': 6.65e-06, 'token_acc': 0.61613974, 'epoch': 0.43, 'global_step/max_steps': '40/94', 'percentage': '42.55%', 'elapsed_time': '1m 18s', 'remaining_time': '1m 45s', 'memory(GiB)': 6.5, 'train_speed(iter/s)': 0.512302}
{'loss': 1.40159426, 'grad_norm': 9.4375, 'learning_rate': 5.79e-06, 'token_acc': 0.64041773, 'epoch': 0.48, 'global_step/max_steps': '45/94', 'percentage': '47.87%', 'elapsed_time': '1m 27s', 'remaining_time': '1m 35s', 'memory(GiB)': 6.5, 'train_speed(iter/s)': 0.512983}
{'loss': 1.54977188, 'grad_norm': 11.9375, 'learning_rate': 4.91e-06, 'token_acc': 0.61078816, 'epoch': 0.53, 'global_step/max_steps': '50/94', 'percentage': '53.19%', 'elapsed_time': '1m 37s', 'remaining_time': '1m 25s', 'memory(GiB)': 6.5, 'train_speed(iter/s)': 0.514489}
{'loss': 1.6754509, 'grad_norm': 13.0625, 'learning_rate': 4.04e-06, 'token_acc': 0.58574393, 'epoch': 0.59, 'global_step/max_steps': '55/94', 'percentage': '58.51%', 'elapsed_time': '1m 46s', 'remaining_time': '1m 15s', 'memory(GiB)': 6.5, 'train_speed(iter/s)': 0.515752}
{'loss': 1.37204351, 'grad_norm': 9.25, 'learning_rate': 3.19e-06, 'token_acc': 0.6391937, 'epoch': 0.64, 'global_step/max_steps': '60/94', 'percentage': '63.83%', 'elapsed_time': '1m 56s', 'remaining_time': '1m 5s', 'memory(GiB)': 6.5, 'train_speed(iter/s)': 0.516829}
{'loss': 1.47697926, 'grad_norm': 11.375, 'learning_rate': 2.4e-06, 'token_acc': 0.62817259, 'epoch': 0.69, 'global_step/max_steps': '65/94', 'percentage': '69.15%', 'elapsed_time': '2m 5s', 'remaining_time': '55s', 'memory(GiB)': 6.5, 'train_speed(iter/s)': 0.517947}
{'loss': 1.4336628, 'grad_norm': 8.125, 'learning_rate': 1.69e-06, 'token_acc': 0.63453862, 'epoch': 0.75, 'global_step/max_steps': '70/94', 'percentage': '74.47%', 'elapsed_time': '2m 14s', 'remaining_time': '46s', 'memory(GiB)': 6.5, 'train_speed(iter/s)': 0.518833}
{'loss': 1.54315252, 'grad_norm': 9.625, 'learning_rate': 1.08e-06, 'token_acc': 0.60202073, 'epoch': 0.8, 'global_step/max_steps': '75/94', 'percentage': '79.79%', 'elapsed_time': '2m 24s', 'remaining_time': '36s', 'memory(GiB)': 6.5, 'train_speed(iter/s)': 0.519627}
{'loss': 1.47180223, 'grad_norm': 9.5625, 'learning_rate': 6e-07, 'token_acc': 0.62211501, 'epoch': 0.85, 'global_step/max_steps': '80/94', 'percentage': '85.11%', 'elapsed_time': '2m 33s', 'remaining_time': '26s', 'memory(GiB)': 6.5, 'train_speed(iter/s)': 0.520284}
{'loss': 1.44068375, 'grad_norm': 10.125, 'learning_rate': 2.5e-07, 'token_acc': 0.62673112, 'epoch': 0.91, 'global_step/max_steps': '85/94', 'percentage': '90.43%', 'elapsed_time': '2m 43s', 'remaining_time': '17s', 'memory(GiB)': 6.5, 'train_speed(iter/s)': 0.520331}
{'loss': 1.44893646, 'grad_norm': 8.375, 'learning_rate': 5e-08, 'token_acc': 0.63837478, 'epoch': 0.96, 'global_step/max_steps': '90/94', 'percentage': '95.74%', 'elapsed_time': '2m 52s', 'remaining_time': '7s', 'memory(GiB)': 6.5, 'train_speed(iter/s)': 0.520707}
{'train_runtime': 183.4332, 'train_samples_per_second': 8.177, 'train_steps_per_second': 0.512, 'train_loss': 1.50650934, 'token_acc': 0.6194337, 'epoch': 1.0, 'global_step/max_steps': '94/94', 'percentage': '100.00%', 'elapsed_time': '3m 3s', 'remaining_time': '0s', 'memory(GiB)': 6.5, 'train_speed(iter/s)': 0.512463}
Train: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 94/94 [03:03<00:00,  1.95s/it]
[INFO:swift] last_model_checkpoint: /workspace/ms-swift/output/v0-20260211-143035/checkpoint-94
[INFO:swift] best_model_checkpoint: None
[INFO:swift] images_dir: /workspace/ms-swift/output/v0-20260211-143035/images
[INFO:swift] End time of running main: 2026-02-11 14:34:09.521336

```
### 3.2. run swift example with Megatron-LM
if you want to use Megatron-LM as Swift's backend, you should set MEGATRON_LM_PATH to /workspace/Megatron-LM-0.15.0 or other versions.

```bash
export MEGATRON_LM_PATH=/workspace/Megatron-LM-0.15.0
cd /workspace/ms-swift
bash examples/megatron/pretrain.sh
```

### 3.3. use other versions of ms-swift
The Metax platform requires the use of MACA-compatible software packages. For instance, compiling depends on torch2.8. We need to use torch2.8+maca3.3.x.x. By default, the installation will overwrite the torch within the environment. Therefore, we recommend using the --no-deps parameter for installation
```bash

git clone -b ${SWIFT_VERSION} https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install . --no-deps

```
After each environment change, the torch and its availability should be checked
```bash
pip list |grep torch
# output:
# torch2.x.x+metax3.x.x.x
```

```python
import torch
torch.cuda.is_available()
```

### 3.4. Differences between Metax and NVIDIA CUDA
We are largely aligned with NVIDIA, but there are some differences in certain software and environment variables.

#### 3.4.1. MACA_MPS_MODE
By default, MACA does not allow multiple processes to run on a single GPU. Therefore, when the GPU is already occupied, you cannot launch another process. To enable this scenario, you need to set MACA_MPS_MODE=1
```bash
# run other scripts ...
export MACA_MPS_MODE=1
cd /workspace/ms-swift/
bash examples/train/full/train.sh
```
#### 3.4.2. MCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME & MCCL_IB_HCA
When using MACA in a multi-node setup, you need to set the environment variables MCCL_SOCKET_IFNAME, GLOO_SOCKET_IFNAME, and MCCL_IB_HCA to ensure proper inter-node communication.
We can use mx-smi and ifconfig to determine which InfiniBand devices and network device are being used.
```bash
ifconfig
# output
ens20f0np0: xxx
            inet: your node ip
            xxx
...
```
```bash
mx-smi topo -n
# output
mx-smi  version: 2.2.9

=================== MetaX System Management Interface Log ===================
Timestamp                                         : Wed Feb 11 18:53:44 2026

Attached GPUs                                     : 8
Device link type matrix
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    NIC0    NIC1    NIC2    NIC3    NIC4    NIC5    Node Affinity  CPU Affinity
GPU0    X       MX      MX      MX      NODE    NODE    NODE    NODE    PIX     PIX     NODE    NODE    SYS     SYS     0              0-31,64-95
GPU1    MX      X       MX      MX      NODE    NODE    NODE    NODE    PIX     PIX     NODE    NODE    SYS     SYS     0              0-31,64-95
GPU2    MX      MX      X       MX      NODE    NODE    NODE    NODE    PIX     PIX     NODE    NODE    SYS     SYS     0              0-31,64-95
GPU3    MX      MX      MX      X       NODE    NODE    NODE    NODE    PIX     PIX     NODE    NODE    SYS     SYS     0              0-31,64-95
GPU4    NODE    NODE    NODE    NODE    X       MX      MX      MX      NODE    NODE    PIX     PIX     SYS     SYS     0              0-31,64-95
GPU5    NODE    NODE    NODE    NODE    MX      X       MX      MX      NODE    NODE    PIX     PIX     SYS     SYS     0              0-31,64-95
GPU6    NODE    NODE    NODE    NODE    MX      MX      X       MX      NODE    NODE    PIX     PIX     SYS     SYS     0              0-31,64-95
GPU7    NODE    NODE    NODE    NODE    MX      MX      MX      X       NODE    NODE    PIX     PIX     SYS     SYS     0              0-31,64-95
NIC0    PIX     PIX     PIX     PIX     NODE    NODE    NODE    NODE    X       PIX     NODE    NODE    SYS     SYS
NIC1    PIX     PIX     PIX     PIX     NODE    NODE    NODE    NODE    PIX     X       NODE    NODE    SYS     SYS
NIC2    NODE    NODE    NODE    NODE    PIX     PIX     PIX     PIX     NODE    NODE    X       PIX     SYS     SYS
NIC3    NODE    NODE    NODE    NODE    PIX     PIX     PIX     PIX     NODE    NODE    PIX     X       SYS     SYS
NIC4    SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     X       PIX
NIC5    SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     PIX     X

Legend:
  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  MX   = Connection traversing MetaXLink
  ETH  = Connection traversing Eth
  NA   = Connection type is unknown

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1
  NIC2: mlx5_2
  NIC3: mlx5_3
  NIC4: mlx5_4
  NIC5: mlx5_5
# The output shows:
#  1. GPU0 to GPU3 communicate with NIC0 and NIC1, while GPU4 to GPU7 communicate with NIC2 and NIC3
#  2. NIC0 uses ib device:mlx5_0, NIC1 uses ib device:mlx5_1, NIC2 uses ib device:mlx5_2, NIC3 uses ib device:mlx5_3

```
Therefore:
MCCL_SOCKET_IFNAME=ens20f0np0
GLOO_SOCKET_IFNAME=ens20f0np0
MCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3

```bash
# node 1
export MCCL_SOCKET_IFNAME=ens20f0np0
export GLOO_SOCKET_IFNAME=ens20f0np0
export MCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
cd /workspace/ms-swift/
bash examples/train/multi-node/torchrun/train_node1.sh
```

```bash
# node 2
# Update the value of the master_addr parameter in the script.
export MCCL_SOCKET_IFNAME=ens20f0np0
export GLOO_SOCKET_IFNAME=ens20f0np0
export MCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
cd /workspace/ms-swift/
bash examples/train/multi-node/torchrun/train_node2.sh
```
