# RL Project — Deep Q-Network from Scratch

This project is my endeavor into reinforcement learning algorithms, consisting of studying and implementating algorithms by scratch alongside my graduate mathematics coursework at UW Madison. The goal is to demonstrate my ability to learn a subject on my own and take a problem from start to finish: understanding the theory, implementing algorithms by hand, running experiments in Gymnasium environments, and documenting reproducible results.

---

## Project Goal

Implement and train a **Deep Q-Network (DQN)** in PyTorch to solve classic control tasks from [Gymnasium](https://gymnasium.farama.org/):

- **CartPole-v1** 
- **LunarLander-v2**   

---

## Success Criteria

- **CartPole-v1:** rolling average reward of at least 475 over the last 100 episodes.  
- **LunarLander-v2:** demonstrate clear improvement over a random baseline, even if not fully solved.  
- **Documentation:** reproducible repo, with logs, plots, and a clear report.  

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
- Rolling average reward (last 100 episodes).  
- Exploration rate and total steps.   

**Logging format:** CSV in `results/` with filename convention:  
results/{env}\_{agent}\_{date}\_{seed}.csv

**Figures:**  
- Reward vs. episode curve.  
- Rolling average vs. episode.  
- Comparison of algorithms (baseline vs. DQN vs. improvements).  

## Reproducibility:

- Random seeds logged for every run.  
- Results stored in `results/` 
- Training scripts runnable from CLI with flags for episodes, seed, and hyperparameters.  
- Plots generated from logs via notebooks/scripts.  

---

## Checklist

- [x] Repo scaffold & README  
- [x] Random baseline agent on CartPole
- [X] Finish Chapters 1-8 in Zhao's "Mathematical Foundations of Reinforcement Learning" 
- [x] Tabular Q-learning (on-policy) on FrozenLake 
- [ ] DQN on CartPole  
- [ ] DQN LunarLander attempt  
- [ ] Final plots & write-up 

