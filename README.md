# 🧠 Q-Learn Lab

### Interactive Reinforcement Learning Tic-Tac-Toe Laboratory

> **Can an agent learn to play Tic-Tac-Toe without being taught a strategy?**

**Q-Learn Lab** is an interactive Reinforcement Learning laboratory that demonstrates how a tabular Q-learning agent can learn Tic-Tac-Toe strategy entirely through **self-play**.

The agent starts with an **empty Q-table and no learned strategy**. It repeatedly plays games, receives rewards, updates its Q-values, and gradually develops a strategy through experience.

This project is designed not just to play Tic-Tac-Toe, but to **visualize and experiment with the learning process itself.**

---

## 🎯 What This Project Demonstrates

The project demonstrates a complete Reinforcement Learning workflow:

```text
              Empty Q-Table
                    │
                    ▼
             Random Exploration
                    │
                    ▼
               Self-Play
                    │
                    ▼
             Reward Signals
                    │
                    ▼
             Q-Value Updates
                    │
                    ▼
          Learned State-Action Values
                    │
                    ▼
             Greedy Evaluation
                    │
                    ▼
           Human vs AI Gameplay
```

The agent is **not given a Tic-Tac-Toe strategy**.

It only knows:

* The rules of the game
* Which moves are legal
* The reward received after actions
* Its current Q-values

Everything else is learned through experience.

---

## ✨ Features

### 🧠 Reinforcement Learning

* Tabular Q-learning
* Self-play training
* ε-greedy exploration
* Q-value updates
* Configurable learning parameters
* Fresh agent for every experiment

### 📈 Training Laboratory

Visualize:

* Training progress
* Reward over games
* Epsilon decay
* Number of learned states
* Training statistics

### 📊 Agent Evaluation

Evaluate the trained agent against a Random Agent and measure:

* Wins
* Draws
* Losses
* Win rate
* Draw rate
* Loss rate

Evaluation uses a **greedy policy** and does not modify the Q-table.

### 🔬 Experiment Tracking

Every training experiment is recorded so different runs can be compared.

Track:

* Training games
* Evaluation games
* Learned states
* Final epsilon
* Win rate
* Draw rate
* Loss rate
* Experiment timestamp

### 🎮 Human vs AI

Play Tic-Tac-Toe against the currently trained agent.

The agent uses its learned Q-table and a greedy policy during gameplay.

Human gameplay does **not** update or retrain the model.

---

## 🏗️ Architecture

```text
reinforcement-learning-tictactoe/
│
├── app.py
│
├── environment/
│   ├── __init__.py
│   └── tictactoe.py
│
├── agents/
│   ├── __init__.py
│   ├── q_learning.py
│   └── random_agent.py
│
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   └── metrics.py
│
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py
│
├── persistence/
│   ├── __init__.py
│   └── model_manager.py
│
├── experiments/
│   └── run_experiments.py
│
├── data/
│   └── experiment_history.json
│
├── models/
│
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore
```

### Component responsibilities

| Component      | Responsibility                 |
| -------------- | ------------------------------ |
| `environment/` | Tic-Tac-Toe game environment   |
| `agents/`      | RL and baseline agents         |
| `training/`    | Self-play training and metrics |
| `evaluation/`  | Measuring agent performance    |
| `persistence/` | Experiment history             |
| `experiments/` | Complete experiment pipeline   |
| `app.py`       | Streamlit interface            |

The architecture intentionally separates the **RL logic from the user interface**.

---

## 🔄 Experiment Pipeline

Each experiment starts from scratch:

```text
Create Fresh Agent
       ↓
Empty Q-Table
       ↓
Self-Play Training
       ↓
Record Training Metrics
       ↓
Evaluate Against Random Agent
       ↓
Save Experiment Results
       ↓
Display Results
```

The Q-table itself is **not permanently persisted**.

Experiment history is stored separately so previous experiments can still be compared.

---

## 📊 Example Experiment

One training run produced:

| Metric           |    Result |
| ---------------- | --------: |
| Training games   |    10,000 |
| Evaluation games |     1,000 |
| Learned states   |     2,226 |
| Final epsilon    |      0.05 |
| Wins             |       780 |
| Draws            |        42 |
| Losses           |       178 |
| Win rate         | **78.0%** |
| Draw rate        |      4.2% |
| Loss rate        |     17.8% |

These results are **one stochastic experiment**, not a claim that the agent will always achieve the same performance.

Different runs can produce different results because the learning process involves randomness.

---

## 🧪 Why Multiple Experiments?

One of the main purposes of this project is experimentation.

For example:

```text
100 games
    ↓
1,000 games
    ↓
10,000 games
    ↓
100,000 games
```

We can investigate questions such as:

* Does more training improve performance?
* How quickly does the Q-table grow?
* When does the number of discovered states begin to plateau?
* How does exploration affect learning?
* How consistent are results across independent runs?
* What happens when learning parameters change?

Instead of presenting a single accuracy number, the project makes the **learning process observable**.

---

## ⚙️ Q-Learning

The agent uses the standard Q-learning update:

```text
Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') − Q(s,a)]
```

Where:

* `s` = current state
* `a` = selected action
* `r` = reward
* `s'` = next state
* `α` = learning rate
* `γ` = discount factor

Action selection uses an ε-greedy strategy.

Early in training:

```text
High ε
   ↓
More exploration
   ↓
More random actions
```

Later in training:

```text
Low ε
   ↓
More exploitation
   ↓
More use of learned Q-values
```

---

## 🖥️ Tech Stack

* **Python**
* **Streamlit**
* **NumPy**
* **Pandas**
* **Plotly**
* **JSON**
* **Git / GitHub**

The core RL algorithm is implemented from scratch rather than relying on an RL library.

---

## 🚀 Running Locally

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd reinforcement-learning-tictactoe
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🧪 Running an Experiment Without the UI

You can also run the complete experiment pipeline directly:

```bash
python -m experiments.run_experiments
```

This will:

1. Create a fresh Q-learning agent
2. Train it through self-play
3. Evaluate it against the Random Agent
4. Save the experiment results
5. Print the results to the terminal

---

## 📌 Important Design Decisions

### The agent starts from zero

Every new experiment creates a new `QLearningAgent`.

Previous training is not carried into the next experiment.

### Human gameplay does not train the model

The Human vs AI mode uses the trained policy but does not update the Q-table.

This keeps gameplay separate from controlled experiments.

### Evaluation is separate from training

The evaluation phase uses a greedy policy and does not perform Q-learning updates.

### Experiment history is persistent

Results from previous experiments remain available even when the current agent is reset.

### The project measures real outcomes

There is no artificial metric such as:

> "Agent intelligence: 87%"

Performance is represented using measurable outcomes such as win rate, draw rate, loss rate, learned states, and training progression.

---

## 🗺️ Future Improvements

Potential extensions include:

* More RL algorithms
* Hyperparameter experimentation
* Multiple opponent strategies
* State-space analysis
* Policy visualization
* Reward analysis
* Larger board configurations
* Neural-network-based agents
* Deep Q-learning
* More advanced self-play environments

The long-term goal is to turn the project into a broader **Reinforcement Learning experimentation platform**.

---

## 📚 Learning Objective

This project was built to understand Reinforcement Learning beyond simply implementing an algorithm.

The laboratory focuses on the complete cycle:

```text
Environment
     ↓
Agent
     ↓
Interaction
     ↓
Experience
     ↓
Learning
     ↓
Evaluation
     ↓
Experimentation
     ↓
Visualization
```

The goal is to make the usually invisible learning process **observable, measurable, and interactive**.

---

## 👨‍💻 Project

**Q-Learn Lab — Reinforcement Learning Tic-Tac-Toe Laboratory**

Built with Python and Streamlit.
