#!/usr/bin/env python
"""
Elo Ranking as SRE Infrastructure (Slide 4)
-------------------------------------------
Treat a dating app's user base as a network of nodes.
- Match = successful transaction
- Dislike = latency / timeout

High-Elo nodes = High-Priority Nodes; exponential score adjustment when
they interact with Low-Priority nodes. A Load Balancer prevents
convergence into a single Super-Node.
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass, field

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_NODES = 50
INITIAL_ELO = 1500
K_BASE = 32
K_EXPONENT = 1.4  # When high-elo vs low-elo, K scales exponentially
LOAD_BALANCER_WEIGHT = 0.6  # 0 = pure merit (top nodes chosen more), 1 = full balance (all nodes equally)
MATCH_PROBABILITY_BASE = 0.5  # Base probability of Match (success) vs Dislike (timeout)
ENTROPY_HISTORY_LEN = 100
LOG_ENTRIES = 12
TICK_INTERVAL = 0.35
NUM_TICKS = 120

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class Node:
    id: int
    elo: float
    matches: int = 0
    dislikes: int = 0

    @property
    def total_interactions(self) -> int:
        return self.matches + self.dislikes


def expected_score(elo_a: float, elo_b: float) -> float:
    """Probability that A 'wins' (gets a Match) in standard Elo."""
    return 1.0 / (1.0 + math.pow(10.0, (elo_b - elo_a) / 400.0))


def k_factor(elo_high: float, elo_low: float) -> float:
    """Exponential K when High-Priority (high-elo) interacts with Low-Priority (low-elo)."""
    diff = elo_high - elo_low
    if diff <= 0:
        return K_BASE
    # Bigger gap -> bigger K (exponential adjustment)
    scale = math.pow(1.0 + diff / 400.0, K_EXPONENT)
    return min(K_BASE * scale, 128.0)


def run_transaction(
    nodes: list[Node],
    idx_a: int,
    idx_b: int,
    rng: random.Random,
) -> tuple[str, float, float]:
    """
    Run one transaction between node A and node B.
    Returns (outcome, delta_a, delta_b) where outcome is "Match" or "Dislike".
    """
    a, b = nodes[idx_a], nodes[idx_b]
    elo_a, elo_b = a.elo, b.elo

    # Who is "high" and "low" for SRE semantics
    if elo_a >= elo_b:
        high_elo, low_elo = elo_a, elo_b
        high_idx, low_idx = idx_a, idx_b
    else:
        high_elo, low_elo = elo_b, elo_a
        high_idx, low_idx = idx_b, idx_a

    e_a = expected_score(elo_a, elo_b)
    e_b = 1.0 - e_a
    k = k_factor(high_elo, low_elo)

    # Outcome: Match (success) or Dislike (timeout)
    is_match = rng.random() < MATCH_PROBABILITY_BASE
    if is_match:
        # Both "win" in a Match (successful transaction)
        s_a, s_b = 1.0, 1.0
        a.matches += 1
        b.matches += 1
        outcome = "Match"
    else:
        # Dislike: low-priority node "loses" more (timeout/latency blame)
        s_a, s_b = 0.0, 0.0
        a.dislikes += 1
        b.dislikes += 1
        # Extra penalty for lower-elo node (exponential SRE edge)
        if high_idx == idx_a:
            s_a, s_b = 0.3, 0.0  # High gets small loss, low gets full loss
        else:
            s_a, s_b = 0.0, 0.3
        outcome = "Dislike"

    delta_a = k * (s_a - e_a)
    delta_b = k * (s_b - e_b)
    a.elo += delta_a
    b.elo += delta_b

    return outcome, delta_a, delta_b


def load_balancer_select(
    nodes: list[Node],
    rng: random.Random,
    weight: float,
) -> tuple[int, int]:
    """
    Select two distinct nodes for a transaction.
    weight=0 -> prefer high-elo (merit). weight=1 -> uniform (balance).
    """
    n = len(nodes)
    if n < 2:
        return 0, 0

    # Weights: blend between "by elo" (high elo more likely) and "uniform"
    elos = [max(100, n.elo) for n in nodes]
    total_elo = sum(elos)
    if total_elo <= 0:
        probs = [1.0 / n] * n
    else:
        merit = [e / total_elo for e in elos]
        uniform = [1.0 / n] * n
        probs = [
            (1.0 - weight) * m + weight * u
            for m, u in zip(merit, uniform)
        ]
    # Normalize
    s = sum(probs)
    probs = [p / s for p in probs]

    idx_a = rng.choices(range(n), weights=probs, k=1)[0]
    idx_b = rng.choices(range(n), weights=probs, k=1)[0]
    while idx_b == idx_a:
        idx_b = rng.choices(range(n), weights=probs, k=1)[0]

    return idx_a, idx_b


def entropy(nodes: list[Node]) -> float:
    """Shannon entropy of the elo distribution (normalized to [0,1] scale)."""
    if not nodes:
        return 0.0
    elos = [max(1e-6, n.elo) for n in nodes]
    total = sum(elos)
    probs = [e / total for e in elos]
    h = -sum(p * math.log2(p) for p in probs if p > 0)
    max_h = math.log2(len(nodes))
    return h / max_h if max_h > 0 else 0.0


def build_layout(
    nodes: list[Node],
    entropy_val: float,
    log_lines: deque[tuple[str, float, float, int, int]],
) -> Layout:
    """Build Rich layout: Top 10 table, entropy panel, transaction log."""
    layout = Layout()

    # ---- Top 10 table ----
    table = Table(title="Top 10 Profiles (Nodes)", show_header=True, header_style="bold cyan")
    table.add_column("Rank", style="dim", width=6)
    table.add_column("Node ID", width=10)
    table.add_column("Elo", justify="right", width=10)
    table.add_column("Matches", justify="right", width=8)
    table.add_column("Dislikes", justify="right", width=8)

    sorted_nodes = sorted(nodes, key=lambda n: n.elo, reverse=True)[:10]
    for i, node in enumerate(sorted_nodes, 1):
        table.add_row(
            str(i),
            f"node_{node.id}",
            f"{node.elo:.1f}",
            str(node.matches),
            str(node.dislikes),
        )

    layout.split_column(
        Layout(name="top", size=14),
        Layout(name="mid", size=6),
        Layout(name="log"),
    )
    layout["top"].update(Panel(table, border_style="green"))

    # ---- Entropy ----
    bar_len = 30
    filled = int(entropy_val * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    entropy_text = Text()
    entropy_text.append("System Entropy: ", style="bold")
    entropy_text.append(f"{entropy_val:.3f} ", style="cyan")
    entropy_text.append(f"[{bar}]", style="dim")
    layout["mid"].update(Panel(entropy_text, title="SRE Health", border_style="yellow"))

    # ---- Transaction log ----
    log_table = Table(show_header=True, header_style="bold magenta", box=None)
    log_table.add_column("Outcome", width=8)
    log_table.add_column("Δ A", width=10)
    log_table.add_column("Δ B", width=10)
    log_table.add_column("Nodes", width=20)
    for outcome, d_a, d_b, ia, ib in list(log_lines)[-LOG_ENTRIES:]:
        style = "green" if outcome == "Match" else "red"
        log_table.add_row(
            outcome,
            f"{d_a:+.1f}",
            f"{d_b:+.1f}",
            f"node_{ia} ↔ node_{ib}",
            style=style,
        )
    layout["log"].update(Panel(log_table, title="Live transaction log", border_style="blue"))

    return layout


def main() -> None:
    console = Console()
    rng = random.Random(42)

    nodes: list[Node] = [
        Node(id=i, elo=INITIAL_ELO + rng.gauss(0, 80))
        for i in range(NUM_NODES)
    ]
    log_lines: deque[tuple[str, float, float, int, int]] = deque(maxlen=LOG_ENTRIES * 2)
    entropy_history: deque[float] = deque(maxlen=ENTROPY_HISTORY_LEN)

    console.print(
        "[bold magenta]Elo Ranking as SRE Infrastructure[/] — "
        "Match = success, Dislike = latency. High-Elo = High-Priority.[/]\n"
    )

    with Live(
        build_layout(nodes, entropy(nodes), log_lines),
        console=console,
        refresh_per_second=4,
        transient=False,
    ) as live:
        for tick in range(NUM_TICKS):
            idx_a, idx_b = load_balancer_select(nodes, rng, LOAD_BALANCER_WEIGHT)
            outcome, d_a, d_b = run_transaction(nodes, idx_a, idx_b, rng)
            log_lines.append((outcome, d_a, d_b, idx_a, idx_b))

            e = entropy(nodes)
            entropy_history.append(e)

            live.update(build_layout(nodes, e, log_lines))
            time.sleep(TICK_INTERVAL)

    console.print("\n[dim]Simulation finished.[/]")


if __name__ == "__main__":
    main()
