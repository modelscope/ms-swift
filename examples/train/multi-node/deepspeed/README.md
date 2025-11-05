# How to run

## 1. Install pdsh in your nodes

```shell
# https://code.google.com/archive/p/pdsh/downloads
# For example, download to /root:
cd /root
wget https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/pdsh/pdsh-2.29.tar.bz2
tar -xvf pdsh-2.29.tar.bz2
cd pdsh-2.29
./configure --prefix=/root/pdsh-2.29 --with-ssh --without-rsh --with-exec --with-timeout=60 --with-nodeupdown --with-rcmd-rank-list=ssh
make
make install
```

In case of the privilege is correct:
```shell
chown root:root /root/pdsh-2.29
```

## Configure the ssh

vim your ~/.ssh/config and input:
```text
Host worker-0
    HostName your-worker-0-ip-here
    User root
Host worker-1
    HostName your-worker-1-ip-here
    User root
```
Say you have two nodes, when doing this, make sure your other nodes can be logined with `ssh root@worker-x` without password(with ssh-key).

## Clone swift repo and run

```shell
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
# If your node number is different, edit examples/train/multi-node/deepspeed/host.txt
sh examples/train/multi-node/deepspeed/train.sh
```
