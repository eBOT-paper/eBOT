# limit
sudo tc qdisc add dev ens3 root tbf rate 100mbit burst 32kbit latency 400ms

# remove
sudo tc qdisc del dev ens3 root
sudo tc qdisc del dev ens3 ingress
