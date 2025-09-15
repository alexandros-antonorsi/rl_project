# RL Keystone Project — Deep Q-Network from Scratch

This project is my **keystone reinforcement learning project**, built independently alongside graduate coursework in mathematics. The goal is to demonstrate my ability to take a problem from start to finish: understanding the theory, implementing algorithms from scratch, running experiments in Gymnasium environments, and documenting reproducible results.

---

## Project Goal

Implement and train a **Deep Q-Network (DQN)** in PyTorch to solve classic control tasks from [Gymnasium](https://gymnasium.farama.org/):

- **CartPole-v1** 
- **LunarLander-v2**  
- Experiment with **stability improvements** (Double DQN, Prioritized Experience Replay)  

---

## Success Criteria

- **CartPole-v1:** rolling average reward ≥475 over last 100 episodes.  
- **LunarLander-v2:** demonstrate clear improvement over random baseline, even if not fully solved.  
- **Documentation:** reproducible repo, with logs, plots, and a clear report.  

---

## Repo Structure
.
├── src/
│ ├── agents/ # simple baseline agents 
│ └── dqn/ # DQN implementation 
├── scripts/ # runnable training/evaluation scripts
├── notebooks/ # analysis/plotting notebooks 
├── tests/ # unit tests for components
├── results/ # training logs, checkpoints, figures
├── README.md # 
├── requirements.txt # Python dependencies
└── .gitignore # 

## Roadmap & Milestones

- **Weeks 2–3 (Sep 15–28):**  
  - Repo scaffold, README, logging conventions.  
  - Random agent baseline script.  

- **Weeks 4–5 (Sep 29–Oct 12):**  
  - Optional tabular Q-learning on FrozenLake.  

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
  - Tag repo release `v1.0`.  

---

## Evaluation & Logging

All training runs will record:
- Episode number, total reward, episode length.  
- Rolling average reward (last 100 episodes).  
- Exploration rate (ε) and total steps.  
- Optional: loss values per update.  

**Logging format:** CSV in `results/` with filename convention:  
results/{env}{agent}{date}_{seed}.csv

**Figures:**  
- Reward vs. episode curve.  
- Rolling average vs. episode.  
- Comparison of algorithms (baseline vs. DQN vs. improvements).  

