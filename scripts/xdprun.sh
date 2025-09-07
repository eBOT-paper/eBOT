#!/bin/bash

filedir=`dirname $0`
filename=eBOT/agg_xdp
ifnet=$1

# install xdp program
sudo $filedir/../eBOT/agg_xdp_usr
#echo sudo xdp-loader load $ifnet $filedir/../$filename.o
#sudo xdp-loader load $ifnet $filedir/../$filename.o
#echo sudo ip link set dev $ifnet xdp obj $filedir/../$filename.o sec xdp/aggregator
#sudo ip link set dev $ifnet xdp obj $filedir/../$filename.o sec xdp/aggregator

# show program
#ip -s link show dev $ifnet
#bpftool prog show
