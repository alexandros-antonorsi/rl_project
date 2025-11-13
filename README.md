# RL Project — Deep Q-Network from Scratch

This project is my endeavor into reinforcement learning algorithms and understanding both their theory and implementation, including how deep neural networks can be integrated to train an agent. The goal is to demonstrate my ability to explore a subject on my own and tackle a problem from start to finish: learning the theory, implementing algorithms by hand, training my agent in Gymnasium environments, and documenting reproducible results. This project was constructed simultaneously alongside my graduate studies in math at UW Madison.

---

## Project Goal

Implement and train a **Deep Q-Network (DQN)** in PyTorch to solve the classic control Gymnasium environment [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/).


## Success Criteria

- **Learning:** understand the theory of reinforcement learning algorithms as well as basic implementation of PyTorch neural networks
- **Implementation:** implement Deep Q Learning on CartPole using a PyTorch deep neural network for the Q function
- **Results:** rolling average reward of at least 475 over the last 100 episodes on CartPole-v1.   
- **Documentation:** reproducible repo with logs, plots, and a clean report.  


---

## Repo Structure
```
├── src/
│ ├── agents/ # training agent class files
│ └── dqn/ # network/buffer for dqn
├── scripts/ # runnable training/evaluation scripts
├── docs/ # final report, plots/figures, and sample logs
├── tests/ # unit tests for components
├── results/ # training logs, checkpoints (gitignored)
├── README.md # project description and instructions 
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
- [ ] PyTorch basics
- [ ] DQN on CartPole   
- [ ] Final plots & write-up 

