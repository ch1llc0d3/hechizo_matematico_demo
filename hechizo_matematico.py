#!/usr/bin/env python
"""
Mathematical Witchcraft Demo (El Hechizo Matemático)
----------------------------------------------------
Compare two profiles in a shared latent space using cosine similarity
and visualize the angle between them in 3D.
"""

from __future__ import annotations

import math
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from sklearn.metrics.pairwise import cosine_similarity


class Ansi:
    """ANSI escape codes for styling console output."""

    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"


FEATURE_SPACE = {
    "tech": 0,
    "indie_hacking": 0,
    "python": 0,
    "cafe": 1,
    "asuncion": 2,
    "near_asuncion": 2,
}

LATENT_DIM = 3


def profile_to_vector(profile: Dict[str, float]) -> np.ndarray:
    """
    Map a sparse profile dict into a dense latent vector R^3.

    Unknown keys are ignored (logged softly).
    Multiple keys mapped to the same latent dimension are combined by max().
    """
    vector = np.zeros(LATENT_DIM, dtype=float)

    for key, value in profile.items():
        idx = FEATURE_SPACE.get(key)
        if idx is None:
            print(
                f"{Ansi.DIM}[latent-mapper] Ignorando feature desconocida: "
                f"'{key}'{Ansi.RESET}"
            )
            continue
        vector[idx] = max(vector[idx], float(value))

    return vector


def detect_and_sanitize_vector(
    vector: np.ndarray,
    label: str,
    max_magnitude: float = 1.5,
    max_l2_norm: float = 5.0,
) -> np.ndarray:
    """
    Lightweight guardrail to detect "vector injection" or extreme outliers.
    """
    v = vector.astype(float).copy()
    flags: List[str] = []

    too_large = np.abs(v) > max_magnitude
    if np.any(too_large):
        flags.append("component_out_of_range")
        v = np.clip(v, -max_magnitude, max_magnitude)

    l2 = np.linalg.norm(v)
    if l2 > max_l2_norm:
        flags.append("l2_out_of_range")
        v = v / l2 * max_l2_norm

    if flags:
        joined = ", ".join(flags)
        print(
            f"{Ansi.RED}{Ansi.BOLD}[security] Vector Injection sospechoso en "
            f"{label}: {joined}. "
            f"Aplicando saneamiento defensivo.{Ansi.RESET}"
        )
    else:
        print(
            f"{Ansi.GREEN}{Ansi.DIM}[security] Vector '{label}' dentro de "
            f"parámetros seguros.{Ansi.RESET}"
        )

    return v


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors using scikit-learn.
    """
    a_2d = vec_a.reshape(1, -1)
    b_2d = vec_b.reshape(1, -1)
    sim = float(cosine_similarity(a_2d, b_2d)[0, 0])
    return sim


def similarity_to_match_probability(similarity: float) -> float:
    """
    Map cosine similarity to "Probabilidad de Amarre" in [0, 100].
    """
    s = max(0.0, min(1.0, similarity))
    boosted = s**1.3
    return boosted * 100.0


def _unit_vector(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def plot_vectors_3d(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    similarity: float,
    theta_rad: float,
) -> None:
    """
    Render a 3D plot of two vectors from the origin.
    """
    fig = plt.figure(figsize=(8, 8))
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    ax.set_facecolor("#050814")
    fig.patch.set_facecolor("#050814")

    origin = np.zeros(3)

    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        vec_a[0],
        vec_a[1],
        vec_a[2],
        color="#00FFC8",
        linewidth=2.5,
        arrow_length_ratio=0.1,
        label="Perfil A",
    )
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        vec_b[0],
        vec_b[1],
        vec_b[2],
        color="#FF6AF6",
        linewidth=2.5,
        arrow_length_ratio=0.1,
        label="Perfil B",
    )

    a_u = _unit_vector(vec_a)
    b_u = _unit_vector(vec_b)
    proj = np.dot(b_u, a_u) * a_u
    b_orth = b_u - proj
    if np.linalg.norm(b_orth) < 1e-8:
        b_orth = np.array([0.0, 0.0, 1.0])
    e2 = _unit_vector(b_orth)

    arc_radius = min(np.linalg.norm(vec_a), np.linalg.norm(vec_b)) * 0.6
    t_vals = np.linspace(0.0, theta_rad, 100)
    arc_points = np.array(
        [
            arc_radius * (math.cos(t) * a_u + math.sin(t) * e2)
            for t in t_vals
        ]
    )

    ax.plot(
        arc_points[:, 0],
        arc_points[:, 1],
        arc_points[:, 2],
        color="#FFD166",
        linewidth=2.0,
    )

    theta_deg = math.degrees(theta_rad)
    mid_idx = len(arc_points) // 2
    mid_point = arc_points[mid_idx]

    ax.text(
        mid_point[0],
        mid_point[1],
        mid_point[2],
        f"θ ≈ {theta_deg:.1f}°",
        color="white",
        fontsize=10,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#050814", ec="#FFD166", lw=1),
    )

    if theta_deg < 25:
        ax.text(
            mid_point[0] * 1.1,
            mid_point[1] * 1.1,
            mid_point[2] * 1.1,
            "MATCH INMINENTE",
            color="#00FF88",
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.4", fc="#050814", ec="#00FF88", lw=1.5
            ),
        )

    max_range = max(
        np.max(np.abs(vec_a)),
        np.max(np.abs(vec_b)),
        1.0,
    )
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    ax.set_xlabel("Latent Dim 1 (Tech)", color="white")
    ax.set_ylabel("Latent Dim 2 (Café)", color="white")
    ax.set_zlabel("Latent Dim 3 (Asunción)", color="white")

    ax.legend(
        facecolor="#050814",
        edgecolor="white",
        labelcolor="white",
        loc="upper left",
    )
    ax.set_title(
        "El Hechizo Matemático: Cosine Similarity en 3D",
        color="white",
        pad=20,
    )

    ax.tick_params(colors="white")

    plt.tight_layout()
    plt.show()


def run_demo() -> None:
    """
    Run the demo.
    """
    profile_a = {"tech": 1.0, "cafe": 0.9, "asuncion": 0.8}
    profile_b = {"indie_hacking": 0.95, "python": 0.85, "near_asuncion": 0.88}

    print(
        f"{Ansi.MAGENTA}{Ansi.BOLD}=== El Hechizo Matemático: "
        f"Cosine Matchmaking Engine ==={Ansi.RESET}"
    )

    vec_a = profile_to_vector(profile_a)
    vec_b = profile_to_vector(profile_b)

    vec_a = detect_and_sanitize_vector(vec_a, label="Perfil A")
    vec_b = detect_and_sanitize_vector(vec_b, label="Perfil B")

    sim = compute_cosine_similarity(vec_a, vec_b)
    theta_rad = math.acos(max(-1.0, min(1.0, sim)))
    theta_deg = math.degrees(theta_rad)
    amarre_pct = similarity_to_match_probability(sim)

    print(
        f"{Ansi.CYAN}{Ansi.BOLD}[engine] Cosine Similarity:"
        f"{Ansi.RESET} {sim:.4f}"
    )
    print(
        f"{Ansi.CYAN}{Ansi.BOLD}[engine] Ángulo θ entre perfiles:"
        f"{Ansi.RESET} {theta_deg:.2f}°"
    )
    print(
        f"{Ansi.YELLOW}{Ansi.BOLD}[oráculo] Probabilidad de Amarre:"
        f"{Ansi.RESET} {Ansi.BOLD}{amarre_pct:6.2f}%{Ansi.RESET}"
    )

    if amarre_pct > 80:
        print(
            f"{Ansi.GREEN}{Ansi.BOLD}[estado] MATCH INMINENTE. "
            f"Los vectores ya se están mirando raro en el espacio latente…"
            f"{Ansi.RESET}"
        )
    elif amarre_pct > 50:
        print(
            f"{Ansi.MAGENTA}{Ansi.BOLD}[estado] Alta afinidad. "
            f"Un par de cafés más y el coseno hace el resto."
            f"{Ansi.RESET}"
        )
    else:
        print(
            f"{Ansi.DIM}[estado] Afinidad moderada. Tal vez necesitan "
            f"más dimensiones en común.{Ansi.RESET}"
        )

    plot_vectors_3d(vec_a, vec_b, similarity=sim, theta_rad=theta_rad)


if __name__ == "__main__":
    run_demo()

