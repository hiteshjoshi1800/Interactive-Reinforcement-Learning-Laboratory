from datetime import datetime
from statistics import mean

import plotly.graph_objects as go
import streamlit as st

from agents.q_learning import QLearningAgent
from environment.tictactoe import TicTacToe
from evaluation.evaluator import evaluate
from persistence.model_manager import ModelManager
from training.trainer import train


st.set_page_config(
    page_title="Q-Learn Lab | Reinforcement Learning Tic-Tac-Toe",
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#7dd3fc"
MUTED = "#94a3b8"
PANEL = "#111827"


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{ background: #080b12; color: #e5e7eb; }}
        [data-testid="stHeader"] {{ background: rgba(8, 11, 18, 0.9); }}
        [data-testid="stSidebar"] {{ background: #0d111a; border-right: 1px solid #1f2937; }}
        [data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid #263244;
            border-radius: 10px; padding: 14px; }}
        [data-testid="stMetricLabel"] {{ color: {MUTED}; font-size: 0.72rem;
            letter-spacing: 0.08em; }}
        [data-testid="stMetricValue"] {{ color: #f8fafc; font-size: 1.55rem; }}
        .lab-kicker {{ color: {ACCENT}; font-size: 0.72rem; font-weight: 700;
            letter-spacing: 0.18em; text-transform: uppercase; }}
        .lab-subtitle {{ color: {MUTED}; margin-top: -0.6rem; }}
        .status {{ color: #d1d5db; font-size: 0.82rem; text-align: right; }}
        .status-dot {{ color: {ACCENT}; font-size: 1rem; }}
        .panel {{ background: {PANEL}; border: 1px solid #263244; border-radius: 10px;
            padding: 18px; }}
        .muted {{ color: {MUTED}; font-size: 0.86rem; }}
        .board-cell button {{ height: 92px; font-size: 2rem; font-weight: 700; }}
        div[data-testid="stButton"] button {{ border-radius: 7px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    if "current_agent" not in st.session_state:
        st.session_state.current_agent = QLearningAgent()
    if "current_metrics" not in st.session_state:
        st.session_state.current_metrics = None
    if "current_evaluation" not in st.session_state:
        st.session_state.current_evaluation = None
    if "current_game" not in st.session_state:
        st.session_state.current_game = None
    if "run_number" not in st.session_state:
        st.session_state.run_number = None
    if "reset_message" not in st.session_state:
        st.session_state.reset_message = None


def model_manager() -> ModelManager:
    return ModelManager()


def history_records() -> list[dict]:
    return model_manager().get_history()


def normalize_run(run: dict) -> dict:
    """Support both the original and current experiment history formats."""
    games = run.get("training_games", run.get("games", 0))
    states = run.get("final_states", run.get("states", 0))
    wins = run.get("wins", 0)
    draws = run.get("draws", 0)
    losses = run.get("losses", 0)
    total = wins + draws + losses
    return {
        "run_id": run.get("run_id", 0),
        "games": games,
        "states": states,
        "evaluation_games": run.get("evaluation_games", total),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": run.get("win_rate", wins / total if total else 0),
        "draw_rate": run.get("draw_rate", draws / total if total else 0),
        "loss_rate": run.get("loss_rate", losses / total if total else 0),
        "final_epsilon": run.get("final_epsilon", 0.05),
        "timestamp": run.get("timestamp", ""),
    }


def status_label() -> tuple[str, str]:
    if st.session_state.current_metrics is None:
        return "READY", "No experiment yet"
    return "TRAINED", f"RUN #{st.session_state.run_number}"


def render_header() -> None:
    status, detail = status_label()
    left, right = st.columns([4, 1])
    with left:
        st.title("Interactive Reinforcement Learning Laboratory")
        st.markdown(
            '<div class="lab-subtitle">Watch an agent learn Tic-Tac-Toe from scratch through self-play.</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f'<div class="status"><span class="status-dot">●</span> {status}<br>{detail}</div>',
            unsafe_allow_html=True,
        )
    st.divider()


def train_new_agent(training_games: int, evaluation_games: int) -> None:
    agent = QLearningAgent()
    progress = st.progress(0, text="Preparing self-play experiment...")
    metrics = train(agent, training_games)
    progress.progress(100, text=f"{training_games:,} games completed")
    results = evaluate(agent, evaluation_games)
    run_data = {
        "training_games": training_games,
        "evaluation_games": evaluation_games,
        "final_states": len(agent.q_table),
        "final_epsilon": agent.epsilon,
        **results,
    }
    model_manager().save_run(run_data)
    saved_run = normalize_run(history_records()[-1])
    st.session_state.current_agent = agent
    st.session_state.current_metrics = metrics
    st.session_state.current_evaluation = results
    st.session_state.current_game = None
    st.session_state.run_number = saved_run["run_id"]
    st.session_state.reset_message = None


def reset_agent() -> None:
    st.session_state.current_agent = QLearningAgent()
    st.session_state.current_metrics = None
    st.session_state.current_evaluation = None
    st.session_state.current_game = None
    st.session_state.run_number = None
    st.session_state.reset_message = "Current agent reset. Experiment history is preserved."


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="lab-kicker">EXPERIMENT CONTROL</div>', unsafe_allow_html=True)
        st.header("Control Center")
        training_games = st.selectbox(
            "Training Games",
            [100, 1_000, 10_000, 100_000, 250_000, 500_000, 1_000_000],
            index=2,
            format_func=lambda x: f"{x:,}",
        )
        evaluation_games = st.selectbox(
            "Evaluation Games", [100, 500, 1_000], index=2, format_func=lambda x: f"{x:,}"
        )
        if st.button("▶  TRAIN NEW AGENT", type="primary", use_container_width=True):
            train_new_agent(training_games, evaluation_games)
            st.rerun()
        if st.button("↻  RESET AGENT", use_container_width=True):
            reset_agent()
            st.rerun()
        st.caption("Reset clears the current Q-table but preserves experiment history.")
        with st.expander("MODEL PARAMETERS"):
            agent = st.session_state.current_agent
            parameters = {
                "Learning rate": agent.learning_rate,
                "Discount factor": agent.discount_factor,
                "Initial epsilon": 1.0,
                "Minimum epsilon": agent.epsilon_min,
                "Epsilon decay": agent.epsilon_decay,
            }
            for label, value in parameters.items():
                st.caption(f"{label}  **{value:.3f}**")


def metric_cards(evaluation: dict | None) -> None:
    metrics = st.session_state.current_metrics
    agent = st.session_state.current_agent
    games = len(metrics.games) if metrics else 0
    win_rate = evaluation["win_rate"] if evaluation else 0
    columns = st.columns(4)
    cards = [
        ("TRAINING GAMES", f"{games:,}"),
        ("LEARNED STATES", f"{len(agent.q_table):,}"),
        ("EPSILON", f"{agent.epsilon:.3f}"),
        ("WIN RATE", f"{win_rate:.1%}"),
    ]
    for column, (label, value) in zip(columns, cards):
        with column:
            st.metric(label, value)


def chart_layout(figure: go.Figure, height: int = 350) -> None:
    figure.update_layout(
        template="plotly_dark",
        height=height,
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font={"color": "#cbd5e1"},
        xaxis={"gridcolor": "#263244"},
        yaxis={"gridcolor": "#263244"},
    )
    st.plotly_chart(figure, use_container_width=True)


def render_learning_charts() -> None:
    metrics = st.session_state.current_metrics
    if metrics is None:
        st.info("Train an agent to generate learning data.")
        return
    rewards = metrics.rewards
    window = min(100, len(rewards))
    rolling = [mean(rewards[max(0, i - window + 1): i + 1]) for i in range(len(rewards))]
    reward_figure = go.Figure()
    reward_figure.add_trace(go.Scatter(x=metrics.games, y=rewards, mode="markers",
                                       marker={"color": "#475569", "size": 3}, name="Reward"))
    reward_figure.add_trace(go.Scatter(x=metrics.games, y=rolling, mode="lines",
                                       line={"color": ACCENT, "width": 3},
                                       name=f"{window}-game average"))
    reward_figure.update_layout(title="Reward vs Games", yaxis_title="Final reward",
                                xaxis_title="Training games")
    chart_layout(reward_figure, 400)
    left, right = st.columns(2)
    with left:
        epsilon_figure = go.Figure(go.Scatter(x=metrics.games, y=metrics.epsilon,
                                               mode="lines", line={"color": ACCENT, "width": 2}))
        epsilon_figure.update_layout(title="Epsilon Decay", yaxis_title="Epsilon",
                                     xaxis_title="Training games", yaxis={"range": [0, 1.05]})
        chart_layout(epsilon_figure)
    with right:
        states_figure = go.Figure(go.Scatter(x=metrics.games, y=metrics.states,
                                              mode="lines", line={"color": ACCENT, "width": 2},
                                              fill="tozeroy", fillcolor="rgba(125,211,252,0.08)"))
        states_figure.update_layout(title="Learned States", yaxis_title="States",
                                    xaxis_title="Training games")
        chart_layout(states_figure)


def render_training_lab() -> None:
    st.subheader("Current Experiment")
    metric_cards(st.session_state.current_evaluation)
    st.subheader("Learning Progress")
    render_learning_charts()
    with st.expander("What is happening?"):
        st.markdown(
            "**STATE** → **CHOOSE ACTION** → **PLAY MOVE** → **RECEIVE REWARD** "
            "→ **UPDATE Q-VALUE** → **NEXT STATE**"
        )
        st.caption("Exploration decreases while the Q-table grows through self-play.")
    if st.session_state.reset_message:
        st.success(st.session_state.reset_message)


def render_performance() -> None:
    st.subheader("Agent Performance")
    st.caption("Evaluation against a Random Agent using a greedy policy.")
    results = st.session_state.current_evaluation
    if results is None:
        st.info("Train an agent first to generate an evaluation report.")
        return
    columns = st.columns(3)
    for column, label, count, rate in zip(
        columns, ("WINS", "DRAWS", "LOSSES"),
        (results["wins"], results["draws"], results["losses"]),
        (results["win_rate"], results["draw_rate"], results["loss_rate"]),
    ):
        with column:
            st.metric(label, f"{count:,}", f"{rate:.1%}")
    figure = go.Figure(go.Bar(
        x=["Wins", "Draws", "Losses"],
        y=[results["win_rate"], results["draw_rate"], results["loss_rate"]],
        text=[f"{results[key]:.1%}" for key in ("win_rate", "draw_rate", "loss_rate")],
        textposition="auto", marker_color=[ACCENT, "#64748b", "#475569"],
    ))
    figure.update_layout(title="Evaluation Outcome Rates", yaxis={"tickformat": ".0%"},
                         yaxis_title="Rate")
    chart_layout(figure, 330)
    total = sum(results[key] for key in ("wins", "draws", "losses"))
    st.markdown(
        f'<div class="panel"><b>PERFORMANCE SUMMARY</b><br>'
        f'<span class="muted">The current agent won {results["wins"]:,} of {total:,} '
        "evaluation games against the Random Agent.<br>"
        "Evaluation is performed without exploration and without modifying the Q-table.</span></div>",
        unsafe_allow_html=True,
    )


def render_experiments() -> None:
    st.subheader("Experiment History")
    st.caption("Every training run is recorded so you can compare how experiments behave.")
    runs = [normalize_run(run) for run in history_records()]
    if not runs:
        st.info("No saved experiments yet.")
        return
    table = [
        {
            "Run": run["run_id"], "Games": f'{run["games"]:,}', "States": f'{run["states"]:,}',
            "Wins": run["wins"], "Draws": run["draws"], "Losses": run["losses"],
            "Win %": f'{run["win_rate"]:.1%}',
        }
        for run in runs
    ]
    headers = ("Run", "Games", "States", "Wins", "Draws", "Losses", "Win %")
    header_html = "".join(f"<th>{header}</th>" for header in headers)
    rows_html = "".join(
        "<tr>" + "".join(f"<td>{row[header]}</td>" for header in headers) + "</tr>"
        for row in table
    )
    st.markdown(
        f"""
        <div style="overflow-x:auto;">
        <table style="width:100%; border-collapse:collapse; background:{PANEL};
                      border:1px solid #263244; border-radius:8px; overflow:hidden;">
            <thead><tr style="color:{MUTED}; text-align:left;">{header_html}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        </div>
        <style>
        table td, table th {{ padding:10px 12px; border-bottom:1px solid #263244; }}
        table tbody tr:last-child td {{ border-bottom:0; }}
        table tbody td {{ color:#e5e7eb; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        figure = go.Figure(go.Scatter(
            x=[run["games"] for run in runs], y=[run["win_rate"] for run in runs],
            mode="lines+markers+text", text=[f'Run #{run["run_id"]}' for run in runs],
            textposition="top center", line={"color": ACCENT},
        ))
        figure.update_layout(title="Win Rate vs Training Games", yaxis={"tickformat": ".0%"})
        chart_layout(figure)
    with right:
        figure = go.Figure(go.Scatter(
            x=[run["games"] for run in runs], y=[run["states"] for run in runs],
            mode="lines+markers+text", text=[f'Run #{run["run_id"]}' for run in runs],
            textposition="top center", line={"color": ACCENT},
        ))
        figure.update_layout(title="Learned States vs Training Games")
        chart_layout(figure)
    st.caption("Results can vary between runs because reinforcement learning contains randomness.")
    selected_id = st.selectbox("Run details", [run["run_id"] for run in runs])
    selected = next(run for run in runs if run["run_id"] == selected_id)
    st.markdown(
        f'<div class="panel"><b>RUN #{selected["run_id"]}</b><br>'
        f'<span class="muted">Training: {selected["games"]:,} games<br>'
        f'Evaluation: {selected["evaluation_games"]:,} games<br>'
        f'Learned states: {selected["states"]:,}<br>'
        f'Final epsilon: {selected["final_epsilon"]:.3f}<br>'
        f'Performance: {selected["win_rate"]:.1%} win · {selected["draw_rate"]:.1%} draw · '
        f'{selected["loss_rate"]:.1%} loss</span></div>',
        unsafe_allow_html=True,
    )


def start_new_game() -> None:
    st.session_state.current_game = TicTacToe()


def play_human_move(action: int) -> None:
    game = st.session_state.current_game
    if game is None or game.is_terminal() or action not in game.get_legal_actions():
        return
    game.step(action)
    if not game.is_terminal():
        agent_action = st.session_state.current_agent.choose_greedy_action(
            game.get_state(), game.get_legal_actions()
        )
        game.step(agent_action)


def render_board() -> None:
    game = st.session_state.current_game
    if game is None:
        st.info("Start a new game to play against the trained policy.")
        return
    winner = game.check_winner()
    if winner == 1:
        status = "YOU WIN"
    elif winner == -1:
        status = "AGENT WINS"
    elif game.is_terminal():
        status = "DRAW"
    else:
        status = "YOUR TURN"
    st.markdown(f"### {status}")
    symbols = {1: "X", -1: "O", 0: ""}
    for row in range(3):
        board_columns = st.columns(3)
        for column, col_index in zip(board_columns, range(3)):
            with column:
                cell = row * 3 + col_index
                label = symbols[game.board[cell]] or " "
                if st.button(
                    label,
                    key=f"cell_{cell}",
                    use_container_width=True,
                    disabled=game.is_terminal() or game.board[cell] != 0,
                ):
                    play_human_move(cell)
                    st.rerun()
    if st.button("NEW GAME", type="primary", use_container_width=True):
        start_new_game()
        st.rerun()
    st.caption("You are X  ·  Agent is O")


def render_play() -> None:
    st.subheader("Play Against the Agent")
    st.caption("Test the strategy the agent learned during training.")
    if st.session_state.current_metrics is None:
        st.markdown(
            '<div class="panel"><h3>NO TRAINED AGENT</h3>'
            '<span class="muted">Train an agent first to play against its learned strategy.</span></div>',
            unsafe_allow_html=True,
        )
        return
    left, right = st.columns([1.2, 1])
    with left:
        if st.session_state.current_game is None and st.button("START GAME", type="primary"):
            start_new_game()
            st.rerun()
        render_board()
    with right:
        st.markdown(
            f'<div class="panel"><b>CURRENT AGENT</b><br><br>'
            f'<span class="muted">Status</span><br>TRAINED<br><br>'
            f'<span class="muted">Training games</span><br>{len(st.session_state.current_metrics.games):,}<br><br>'
            f'<span class="muted">Learned states</span><br>{len(st.session_state.current_agent.q_table):,}<br><br>'
            '<span class="muted">Exploration</span><br>OFF<br><br>'
            '<span class="muted">Policy</span><br>GREEDY</div>',
            unsafe_allow_html=True,
        )


def main() -> None:
    initialize_state()
    inject_styles()
    render_sidebar()
    render_header()
    tabs = st.tabs(["Training Lab", "Performance", "Experiments", "Play"])
    with tabs[0]:
        render_training_lab()
    with tabs[1]:
        render_performance()
    with tabs[2]:
        render_experiments()
    with tabs[3]:
        render_play()


if __name__ == "__main__":
    main()
