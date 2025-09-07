#!/bin/bash

### This setup is to install xdp/ebpf environment
### For more information: https://github.com/xdp-project/xdp-tutorial/blob/main/setup_dependencies.org 

echo "### updating packages"
sudo apt update

echo "### install dependencies"
echo "### install 'perf' utility"
echo "### install kernel headers dependencies"
echo "### install extra tools: bpftool, tcpdump"
echo "### install bpf/bpf_helpers" ## we can install from sources at https://github.com/libbpf/libbpf

sudo NEEDRESTART_MODE=a apt-get -y install clang llvm libelf-dev libpcap-dev build-essential libc6-dev-i386 m4 linux-tools-$(uname -r) linux-headers-$(uname -r) linux-tools-common linux-tools-generic tcpdump libbpf-dev

echo "### install libxdp and xdp-tools"
DIR=$HOME/xdp-tools
git clone https://github.com/xdp-project/xdp-tools.git $DIR
## if libbpf is not exist: git -C $DIR submodule init && git submodule update
make -C $DIR
sudo make -C $DIR install
sudo ldconfig
#rm -f $DIR

echo "### install python3 dependancies: pytorch, websockets, python-dotenv"
sudo apt-get -y install python3-pip
sudo pip3 install websockets python-dotenv
sudo pip3 install torch torchvision
