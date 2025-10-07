# RL Keystone Project — Deep Q-Network from Scratch

This project is my endeavor into **reinforcement learning algorithms**, built independently in a single semester alongside my graduate coursework in mathematics at UW Madison. The goal is to demonstrate my ability to learn a subject on my own and take a problem from start to finish: understanding the theory, implementing algorithms from scratch, running experiments in Gymnasium environments, and documenting reproducible results.

---

## Project Goal

Implement and train a **Deep Q-Network (DQN)** in PyTorch to solve classic control tasks from [Gymnasium](https://gymnasium.farama.org/):

- **CartPole-v1** 
- **LunarLander-v2**  
- Experiment with **stability improvements** (Double DQN, Prioritized Experience Replay)  

---

## Success Criteria

- **CartPole-v1:** rolling average reward of at least 475 over the last 100 episodes.  
- **LunarLander-v2:** demonstrate clear improvement over a random baseline, even if not fully solved.  
- **Documentation:** reproducible repo, with logs, plots, and a clear report.  

---

## Repo Structure
```
├── src/
│ ├── agents/ # simple baseline agents 
│ └── dqn/ # DQN implementation 
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
- Optional: loss values per update.  

**Logging format:** CSV in `results/` with filename convention:  
results/{env}\_{agent}\_{date}\_{seed}.csv

**Figures:**  
- Reward vs. episode curve.  
- Rolling average vs. episode.  
- Comparison of algorithms (baseline vs. DQN vs. improvements).  

## Reproducibility:

- Fixed seeds logged for every run.  
- Results stored in `results/` 
- Training scripts runnable from CLI with flags for env, episodes, seed.  
- Plots generated from logs via notebooks/scripts.  

---

## Checklist

- [x] Repo scaffold & README  
- [x] Random baseline agent on CartPole  
- [ ] Tabular Q-learning   
- [ ] DQN on CartPole  
- [ ] Double DQN / Prioritized Replay  
- [ ] LunarLander attempt  
- [ ] Final plots & write-up 

