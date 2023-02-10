# Implementing Reinforcement Learning Datacenter Congestion Control in NVIDIA NICs
RL-CC is an RDMA congestion control algorithm based on reinforcement learning. 
This repository contains a trained RL-CC neural network model and a script to distill it into an ensemble of decision trees using synthetic data. The RL-CC model was trained using a proprietary NVIDIA congestion control simulator and the synthetic data is designed to approximate the data distribution that is created when running the model within the simulator.  

To distill the model:
```cmd
1. pip install -r requirements.txt
2. python3 distill_network.py
```
<!-- The resulting distilled tree ensemble model can be written as a set of if-else conditions, which -->