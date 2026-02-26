#!/usr/bin/env python
"""
Dating App as High-Availability Distributed System (SRE Simulation)
-------------------------------------------------------------------
Treats the user base as a distributed system: each user is a Node, Likes are
Successful Transactions, Dislikes/Ignores are Latency/Timeout events.
Uses Elo rating (Chess / early Tinder-style) with SRE twists: Network Effect
and Load Balancer to prevent Super-Node monopolization.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Outcome: Like = Successful Transaction (match); Dislike/Ignore = Timeout
# ---------------------------------------------------------------------------

class InteractionOutcome(str, Enum):
    """Transaction result between two nodes (SRE view of a dating event)."""
    MATCH = "match"           # Mutual like — successful transaction both ways
    REJECT_A = "reject_a"     # B rejected A — A gets timeout, B gets success
    REJECT_B = "reject_b"     # A rejected B — B gets timeout, A gets success


# ---------------------------------------------------------------------------
# UserNode: Each profile is a node in the distributed system
# ---------------------------------------------------------------------------

@dataclass
class UserNode:
    """
    A single node in the dating graph. Attributes aligned with SRE semantics:
    elo_score = node priority/desirability, connection_history = audit log.
    """
    node_id: str
    elo_score: float = 1500.0
    k_factor: float = 32.0
    connection_history: List[Tuple[str, str, float]] = field(default_factory=list)
    # SRE: track interaction count for load balancing (prevent super-node monopoly)
    interaction_count: int = 0

    def record_interaction(self, peer_id: str, outcome_label: str, delta: float) -> None:
        """Append to connection_history for observability and debugging."""
        self.connection_history.append((peer_id, outcome_label, delta))
        self.interaction_count += 1

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UserNode):
            return False
        return self.node_id == other.node_id


# ---------------------------------------------------------------------------
# Elo: Expected score and vanilla update (pre–network effect)
# ---------------------------------------------------------------------------

def _expected_score(elo_self: float, elo_opponent: float) -> float:
    """Probability that self 'wins' the interaction (Logic for the 150 IQ cluster)."""
    return 1.0 / (1.0 + math.pow(10.0, (elo_opponent - elo_self) / 400.0))


def _vanilla_elo_delta(
    elo_self: float,
    elo_opponent: float,
    k: float,
    actual_score: float,
) -> float:
    """Standard Elo delta: K * (S - E). No network effect yet."""
    expected = _expected_score(elo_self, elo_opponent)
    return k * (actual_score - expected)


# ---------------------------------------------------------------------------
# Network Effect (Maverick): High–Low Elo interaction => exponential impact
# ---------------------------------------------------------------------------

def _network_effect_multiplier(elo_high: float, elo_low: float, scale: float = 400.0) -> float:
    """
    When a High-Priority Node interacts with a low-Elo node, impact is exponential.
    Prevents the system from being too 'cold' — big gaps create bigger ranking shifts.
    """
    gap = max(0.0, elo_high - elo_low)
    # Exponential boost: gap/scale in exponent, clamp to avoid explosion
    return min(math.exp(gap / scale), 3.0)


def process_interaction(
    node_a: UserNode,
    node_b: UserNode,
    outcome: InteractionOutcome,
) -> Tuple[float, float]:
    """
    Process one interaction (transaction) between two nodes. Updates both Elo
    scores. Applies the Network Effect: high–low Elo pairs get exponential
    impact on the ranking delta.
    """
    elo_a, elo_b = node_a.elo_score, node_b.elo_score
    k_a, k_b = node_a.k_factor, node_b.k_factor

    # Actual scores: match = 0.5/0.5; reject_a = 0 for A, 1 for B; reject_b = 1 for A, 0 for B
    if outcome == InteractionOutcome.MATCH:
        s_a, s_b = 0.5, 0.5
    elif outcome == InteractionOutcome.REJECT_A:
        s_a, s_b = 0.0, 1.0
    else:  # REJECT_B
        s_a, s_b = 1.0, 0.0

    # Vanilla deltas
    delta_a = _vanilla_elo_delta(elo_a, elo_b, k_a, s_a)
    delta_b = _vanilla_elo_delta(elo_b, elo_a, k_b, s_b)

    # Maverick: apply network effect — high/low Elo pair => exponential impact, not linear
    elo_high = max(elo_a, elo_b)
    elo_low = min(elo_a, elo_b)
    mult = _network_effect_multiplier(elo_high, elo_low)
    delta_a *= mult
    delta_b *= mult

    # Apply updates
    node_a.elo_score += delta_a
    node_b.elo_score += delta_b

    # Audit log
    node_a.record_interaction(node_b.node_id, outcome.value, delta_a)
    node_b.record_interaction(node_a.node_id, outcome.value, delta_b)

    return delta_a, delta_b


# ---------------------------------------------------------------------------
# Load Balancer: Prevent Super-Nodes from monopolizing all traffic (matches)
# ---------------------------------------------------------------------------

def select_peer_for_request(
    requesting_node: UserNode,
    nodes: List[UserNode],
    traffic_weights: Optional[Dict[str, int]] = None,
    super_node_penalty_factor: float = 0.3,
) -> Optional[UserNode]:
    """
    SRE Load Balancer: when 'requesting_node' needs a match candidate, we don't
    always return the highest-Elo node. We down-weight nodes that have already
    received too much traffic (influencers/super-nodes) so that lower-priority
    nodes also get exposure. Uses traffic_weights (node_id -> request count).
    """
    if not nodes:
        return None
    # Exclude self
    candidates = [n for n in nodes if n.node_id != requesting_node.node_id]
    if not candidates:
        return None

    weights = traffic_weights or defaultdict(int)
    elo_max = max(n.elo_score for n in candidates)

    def effective_weight(node: UserNode) -> float:
        # Base weight: we want some preference for higher Elo (desirability)
        elo_norm = node.elo_score / max(elo_max, 1.0)
        traffic = weights.get(node.node_id, 0)
        # Maverick: penalize super-nodes so they don't monopolize all traffic
        penalty = 1.0 / (1.0 + super_node_penalty_factor * traffic)
        return elo_norm * penalty

    weighted = [(n, effective_weight(n)) for n in candidates]
    total = sum(w for _, w in weighted)
    if total <= 0:
        return random.choice(candidates)
    r = random.uniform(0, total)
    for node, w in weighted:
        r -= w
        if r <= 0:
            return node
    return weighted[-1][0]


def get_traffic_weights(nodes: List[UserNode]) -> Dict[str, int]:
    """Build traffic count per node from connection_history (who got how many requests)."""
    weights: Dict[str, int] = defaultdict(int)
    for node in nodes:
        weights[node.node_id] = node.interaction_count
    return dict(weights)


# ---------------------------------------------------------------------------
# System metrics: Entropy (match diversity), Tiers, Promotion detection
# ---------------------------------------------------------------------------

def system_entropy(nodes: List[UserNode]) -> float:
    """
    Entropy of the match system: how diverse the interactions are.
    Higher = more nodes participating in matches; lower = concentrated on few nodes.
    Uses interaction_count distribution.
    """
    if not nodes:
        return 0.0
    total = sum(n.interaction_count for n in nodes)
    if total <= 0:
        return 0.0
    probs = [n.interaction_count / total for n in nodes if n.interaction_count > 0]
    if not probs:
        return 0.0
    return -sum(p * math.log2(p) for p in probs if p > 0)


def get_top_nodes(nodes: List[UserNode], top_k: int = 10) -> List[UserNode]:
    """Top-K High-Priority Nodes by Elo (desirability)."""
    sorted_nodes = sorted(nodes, key=lambda n: n.elo_score, reverse=True)
    return sorted_nodes[:top_k]


# Tier thresholds (Maverick: Desirability Peak = Tier 1)
TIER_1_ELO = 1800.0
TIER_2_ELO = 1600.0


def check_promotion_to_tier1(node: UserNode, previous_elo: float) -> bool:
    """True if node just crossed into Tier 1 (Desirability Peak)."""
    return previous_elo < TIER_1_ELO <= node.elo_score


def get_tier(elo: float) -> int:
    """1 = top, 2 = mid, 3 = rest."""
    if elo >= TIER_1_ELO:
        return 1
    if elo >= TIER_2_ELO:
        return 2
    return 3


# ---------------------------------------------------------------------------
# Rich Terminal Dashboard (real-time SRE observability)
# ---------------------------------------------------------------------------

def make_dashboard(
    nodes: List[UserNode],
    entropy: float,
    promotion_log: List[str],
    step: int,
) -> Layout:
    """Build the real-time terminal dashboard: Top 10, Entropy, Promotion log."""
    layout = Layout()

    # ---- Top 10 High-Priority Nodes (Top Profiles) ----
    table = Table(
        title="[bold cyan]Top 10 High-Priority Nodes[/] (Desirability Ranking)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Rank", style="dim", width=6)
    table.add_column("Node ID", style="green")
    table.add_column("Elo Score", justify="right", style="yellow")
    table.add_column("Tier", justify="center", width=6)
    table.add_column("Interactions", justify="right", style="dim")
    top = get_top_nodes(nodes, 10)
    for i, node in enumerate(top, 1):
        tier = get_tier(node.elo_score)
        table.add_row(
            str(i),
            node.node_id,
            f"{node.elo_score:.1f}",
            f"T{tier}",
            str(node.interaction_count),
        )
    layout.split_column(
        Layout(name="top", size=16),
        Layout(name="bottom"),
    )
    layout["top"].update(Panel(table, title="[bold]Cluster State[/]", border_style="blue"))

    # ---- System Entropy + Promotion log ----
    entropy_text = Text()
    entropy_text.append("System Entropy (match diversity): ", style="bold")
    entropy_text.append(f"{entropy:.3f}", style="cyan")
    entropy_text.append("  — Higher = more distributed traffic.\n", style="dim")
    log_lines = promotion_log[-12:]  # Last 12 entries
    log_text = "\n".join(log_lines) if log_lines else "No promotions yet."
    layout["bottom"].split_row(
        Layout(Panel(entropy_text, title="[bold]Metrics[/]", border_style="green"), minimum_size=40),
        Layout(
            Panel(
                log_text,
                title="[bold]Promotion Log[/] (Tier 1 = Desirability Peak)",
                border_style="yellow",
            ),
            minimum_size=50,
        ),
    )
    return layout


def run_simulation_with_dashboard(
    num_nodes: int = 30,
    num_steps: int = 200,
    seed: Optional[int] = 42,
) -> None:
    """
    Run the distributed-system simulation with a live Rich dashboard.
    Each step: pick a requesting node, load-balance select a peer, process interaction,
    check for Tier 1 promotion, refresh dashboard.
    """
    if seed is not None:
        random.seed(seed)
    console = Console()

    # Bootstrap node pool (Logic for the 150 IQ cluster: start all at 1500, let dynamics emerge)
    nodes = [
        UserNode(node_id=f"node_{i:03d}", elo_score=1500.0, k_factor=32.0)
        for i in range(num_nodes)
    ]
    promotion_log: List[str] = []

    def one_step(step_id: int) -> None:
        # Load balancer: who is "requesting" a match and who do we show them?
        requester = random.choice(nodes)
        traffic = get_traffic_weights(nodes)
        peer = select_peer_for_request(requester, nodes, traffic_weights=traffic)
        if peer is None:
            return
        # Random outcome (in production this would be real user actions)
        outcome = random.choice(list(InteractionOutcome))
        elo_a_before = requester.elo_score
        elo_b_before = peer.elo_score
        process_interaction(requester, peer, outcome)
        # Promotion detection: Tier 1 = Desirability Peak
        if check_promotion_to_tier1(requester, elo_a_before):
            promotion_log.append(
                f"Node [{requester.node_id}] promoted to Tier 1 - Desirability Peak Detected"
            )
        if check_promotion_to_tier1(peer, elo_b_before):
            promotion_log.append(
                f"Node [{peer.node_id}] promoted to Tier 1 - Desirability Peak Detected"
            )

    with Live(
        make_dashboard(nodes, system_entropy(nodes), promotion_log, 0),
        console=console,
        refresh_per_second=4,
        screen=False,
    ) as live:
        for step in range(1, num_steps + 1):
            one_step(step)
            live.update(
                make_dashboard(nodes, system_entropy(nodes), promotion_log, step)
            )

    console.print("[bold green]Simulation complete.[/] Final state above.")
    return None


if __name__ == "__main__":
    run_simulation_with_dashboard(num_nodes=30, num_steps=200, seed=42)
