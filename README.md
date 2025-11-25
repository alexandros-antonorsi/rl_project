# Reinforcement Learning Project — Deep Q Network w/ PyTorch

This project is my introduction to reinforcement learning and neural networks. My goals for this project were to:
- **Learning:** Learn the theory of RL algorithms from Shiyu Zhao's "Mathematical Foundations of Reinforcement Learning"
- **Implementation:** Implement a DQN agent in PyTorch to solve the classic control Gymnasium environment [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/).
- **Results:** Have the DQN agent achieve max reward (500) over the last 100 episodes on CartPole-v1.   
- **Documentation:** Create reproducible code and a polished final report detailing results.  

## Repo Structure
```
├── agents/ # agent class files
├── scripts/ # training scripts
├── plots/ # final plots/figures
├── logs/ # training logs (gitignored)
├── README.md # project description and writeup
├── requirements.txt # Python dependencies
└── .gitignore # ignore rules (includes .venv, results/, pycache)
```

## Reproducibility:


- All runs logged with .csv filename convention: logs/{env}\_{agent}\_{date}\_{seed}.csv  
- Training scripts runnable from CLI with args for episodes, seed, and hyperparameters. 
- Plots generated from logs via script.

---

