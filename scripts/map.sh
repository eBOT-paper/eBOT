#!/bin/bash

#sudo vim /sys/fs/bpf/ens3/aggregator_map
#sudo bpftool map dump name aggregator_map
sudo bpftool map dump name $1_map
