# RL Project — Deep Q Networks w/ PyTorch

This project is my introduction to Reinforcement Learning (RL) and neural networks. The goal is to learn the theory of RL algorithms, understand the implementation of Deep Q Networks (DQNs) using Pytorch, and creating reproducible code with documented results.

## Success Criteria

- **Learning:** Learned the theory of RL algorithms from reading Shiyu Zhao's "Mathematical Foundations of Reinforcement Learning"
- **Implementation:** Implemented a DQN agent in PyTorch to solve the classic control Gymnasium environment [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/).
- **Results:** DQN agent achieved max reward (500) over the last 100 episodes on CartPole-v1.   
- **Documentation:** Reproducible repo code and a polished final report detailing results.  

## Repo Structure
```
├── agents/ # agent class files
├── scripts/ # training scripts
├── docs/ # final report, plots/figures, and sample logs
├── results/ # training logs (gitignored)
├── README.md  
├── requirements.txt # Python dependencies
└── .gitignore # ignore rules (includes .venv, results/, pycache)
```

## Evaluation & Logging

All training runs will record:
- Episode number, total reward, episode length.  
- Rolling average reward over last 100 episodes.  
- Exploration rate and total steps.   

**Logging format:** CSV in `results/` with filename convention:  
results/{env}\_{agent}\_{date}\_{seed}.csv

**Figures:**  
- Reward vs. episode curve.  
- Rolling average vs. episode.  

## Reproducibility:

- Random seeds logged for every run.  
- Results stored in `results/` 
- Training scripts runnable from CLI with flags for episodes, seed, and hyperparameters.  
- Plots generated from logs via notebooks/scripts.  

---

## Checklist

- [x] Repo scaffold & README  
- [x] Random baseline agent on CartPole
- [X] Read Chapters 1-8 in Zhao's "Mathematical Foundations of Reinforcement Learning" 
- [x] Tabular Q-learning warm-up on FrozenLake 
- [X] PyTorch basics
- [ ] DQN on CartPole   
- [ ] Final plots & write-up 

