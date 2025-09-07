#!/bin/bash

logpath=/var/log/dt-agent-error.log
workdir=/home/ubuntu/fedml-ebpf/ebpf/agg-training
pycmd=/home/ubuntu/miniconda3/envs/python3.9/bin/python
cmd="nohup bash -c 'cd $workdir/ && $pycmd $workdir/agent.py' > $logpath 2>&1 &"

for i in {2..2}; do
    ssh node$i $cmd
    echo "Command started on node$i"
done
