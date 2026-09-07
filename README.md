# Interactive Reinforcement Learning Laboratory 2.0

An interactive Streamlit laboratory for training, evaluating, visualizing, and playing against a reinforcement-learning Tic-Tac-Toe agent.

The project is designed to make the learning process visible—not just the final game. You can inspect reward progression, exploration decay, learned states, experiment history, evaluation results, and the agent's gameplay policy from one dashboard.

## Features

- Tabular Q-learning with epsilon-greedy exploration
- Discount factor configured for long-term strategy
- Board-symmetry updates for rotations and reflections
- Mixed training opponents:
  - Random opponent
  - Tactical rule-based opponent
  - Minimax opponent
  - Self-play
- Interactive training controls up to 1,000,000 games
- Training visualizations:
  - Reward progression
  - Rolling reward average
  - Epsilon decay
  - Learned-state growth
- Evaluation against a random opponent
- Persistent experiment history in `data/experiment_history.json`
- Human-versus-agent Tic-Tac-Toe board
- Hybrid gameplay policy with Minimax tactical safety
- Automated tests for the environment, agent, and training pipeline

## How the agent learns

The agent estimates the value of taking an action in a board state:

```text
State
  ↓
Choose action
  ↓
Play move
  ↓
Receive reward
  ↓
Update Q-value
  ↓
Observe next state
```

The Q-learning update uses:

```text
Q(s, a) ← Q(s, a) + α [r + γ max Q(s', a') − Q(s, a)]
```

Current default model parameters:

| Parameter | Value |
| --- | ---: |
| Learning rate | `0.10` |
| Discount factor | `0.95` |
| Initial epsilon | `1.00` |
| Minimum epsilon | `0.05` |
| Epsilon decay | `0.995` |

The trainer exposes the agent to different opponent styles so it does not learn only against random play. During human gameplay, the learned policy is combined with Minimax tactical protection so the agent can take immediate wins, block immediate losses, and avoid basic tactical mistakes.

## Dashboard tabs

### Training Lab

Start a fresh experiment and inspect:

- Training games
- Learned Q-table states
- Current epsilon
- Evaluation win rate
- Reward versus games
- Epsilon decay
- Learned-state growth

### Performance

View wins, draws, losses, outcome rates, and a summary of the latest evaluation against the random opponent. Evaluation uses a greedy policy and does not modify the Q-table.

### Experiments

Compare saved runs using:

- Training games
- Learned states
- Wins, draws, and losses
- Win, draw, and loss rates
- Final epsilon
- Timestamp

### Play

Play as X against the current trained agent. The board uses the hybrid policy described above. Human gameplay does not retrain the model.

## Project structure

```text
.
├── app.py                         # Streamlit dashboard
├── agents/
│   ├── q_learning.py              # Tabular Q-learning agent
│   ├── random_agent.py            # Random baseline
│   ├── rule_based.py              # Tactical baseline
│   └── minimax.py                 # Perfect tactical opponent
├── environment/
│   └── tictactoe.py               # Game rules and state transitions
├── training/
│   ├── trainer.py                 # Mixed-opponent training loop
│   └── metrics.py                 # Training history container
├── evaluation/
│   └── evaluator.py               # Greedy evaluation
├── persistence/
│   └── model_manager.py           # Experiment history storage
├── experiments/
│   └── run_experiments.py         # Scripted experiment runner
├── data/
│   └── experiment_history.json    # Saved experiment summaries
├── tests/                         # Automated tests
├── requirements.txt
└── README.md
```

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/hiteshjoshi1800/Interactive-Reinforcement-Learning-Laboratory.git
cd Interactive-Reinforcement-Learning-Laboratory
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the dashboard

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Run tests

```bash
python -m pytest -q tests
```

## Training recommendations

Suggested starting points:

| Training games | Use |
| ---: | --- |
| `10,000` | Fast experiment and dashboard demonstration |
| `100,000` | More stable learning |
| `250,000+` | Longer experiments and comparison |
| `1,000,000` | Extended training; can take considerably longer |

More games are not the only way to improve performance. Opponent quality, reward design, exploration settings, symmetry handling, and evaluation methodology also affect results.

## Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/).
2. Sign in with GitHub.
3. Create a new app.
4. Select this repository.
5. Choose branch `master`.
6. Set the main file to `app.py`.
7. Deploy.

Streamlit will install the pinned dependencies from `requirements.txt`.

## Limitations

- The Q-table is tabular and specific to Tic-Tac-Toe.
- Training contains randomness, so runs can produce different results.
- A high win rate against a random opponent does not prove expert-level human performance.
- Experiment summaries are persisted locally in `data/experiment_history.json`.
- Streamlit Cloud may have execution-time limits for very large training runs.

## License

No license has been specified yet.
