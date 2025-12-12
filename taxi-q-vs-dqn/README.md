<h1 align="center">⋆ ˚｡⋆୨ taxi-v3 rl comparison ୧⋆ ˚｡⋆</h1>


<p align="center"><i>⋆ ˚｡⊹˚₊ comparing classical rl and deep rl in a toy environment ₊˚⊹˚｡⋆</i></p>

---

### ⋆౨ৎ˚⟡˖ ࣪ overview  

**taxi-v3 rl comparison** explores how **tabular q-learning** compares to **deep q-learning (dqn)** in a small, discrete environment from **ai gymnasium**.

the project focuses on **learning speed** and **training stability**, demonstrating how algorithm choice depends on environment complexity.

```bash
🚕 environment → taxi-v3 (gymnasium toy text)
🧠 methods → tabular q-learning · deep q-learning (dqn)
📈 comparison → learning speed · stability
🎯 goal → understand when deep rl is necessary vs overkill
```

---

### ⋆౨ৎ˚⟡˖ ࣪ tech stack  

**languages** python  
**libraries** gymnasium · numpy · torch · matplotlib  
**tools** vs code · git · github  

---

### ⋆౨ৎ˚⟡˖ ࣪ methods  

- tabular q-learning → q-table with epsilon-greedy exploration  
- deep q-learning (dqn) → mlp-based network with replay buffer and target network  
- evaluation → reward curves with moving average smoothing  

---

### ⋆౨ৎ˚⟡˖ ࣪ results  

⚡ dqn learns faster during early training  
📉 q-learning is more stable after convergence  
🧠 small discrete environments favor tabular methods  

---

### ⋆౨ৎ˚⟡˖ ࣪ visual preview  

image file: results/learning_curves.png  

---

### ⋆౨ৎ˚⟡˖ ࣪ how to run  

pip install -r requirements.txt  
python train_qlearning.py  
python train_dqn.py  
python plots.py  

---

### ⋆౨ৎ˚⟡˖ ࣪ key takeaway  

simpler reinforcement learning methods can outperform deep rl in structured, low-complexity environments.

---

### ⋆౨ৎ˚⟡˖ ࣪ about
miyah dones  
computer science + molecular biology @ towson university