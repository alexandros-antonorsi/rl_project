# RL Keystone Project — Deep Q-Network from Scratch

This project is my endeavor into **reinforcement learning algorithms**, built independently alongside my graduate coursework in mathematics at UW Madison. The goal is to demonstrate my ability to learn a subject on my own and take a problem from start to finish: understanding the theory, implementing algorithms from scratch, running experiments in Gymnasium environments, and documenting reproducible results.

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
├── notebooks/ # analysis/plotting notebooks 
├── tests/ # unit tests for components
├── results/ # training logs, checkpoints, figures (gitignored)
├── README.md # project description and instructions 
├── requirements.txt # Python dependencies
└── .gitignore # ignore rules (includes .venv, results/, pycache)
```

## Roadmap & Milestones

- **Weeks 2–3 (Sep 15–28):**  
  - Repo scaffold, README, logging conventions.  
  - Random agent baseline script.  

- **Weeks 4–5 (Sep 29–Oct 12):**  
  - Tabular Q-learning on FrozenLake.  

- **Weeks 6–7 (Oct 13–26):**  
  - Linear approximation Q-learning on CartPole.  

- **Weeks 8–10 (Oct 27–Nov 16):**  
  - Core DQN (replay buffer, target network, epsilon schedule).  
  - Solve CartPole.  

- **Weeks 11–12 (Nov 17–30):**  
  - Attempt LunarLander.  
  - Experiment with Double DQN or Prioritized Replay.  

- **Weeks 13–15 (Dec 1–18):**  
  - Save final results, generate plots, polish README + short report.   

---

## Evaluation & Logging

All training runs will record:
- Episode number, total reward, episode length.  
- Rolling average reward (last 100 episodes).  
- Exploration rate and total steps.  
- Optional: loss values per update.  

**Logging format:** CSV in `results/` with filename convention:  
results/{env}{agent}{date}_{seed}.csv

**Figures:**  
- Reward vs. episode curve.  
- Rolling average vs. episode.  
- Comparison of algorithms (baseline vs. DQN vs. improvements).  

## Reproducibility:

- Fixed seeds logged for every run.  
- Results stored in `results/` (not in repo).  
- Training scripts runnable from CLI with flags for env, episodes, seed.  
- Plots generated from logs via notebooks/scripts.  

---

## Checklist

- [x] Repo scaffold & README  
- [ ] Random baseline agent on CartPole  
- [ ] Tabular Q-learning (optional)  
- [ ] Linear approximation agent  
- [ ] DQN on CartPole  
- [ ] Double DQN / Prioritized Replay  
- [ ] LunarLander attempt  
- [ ] Final plots & write-up 

