import math

import numpy as np
import pandas as pd


def add_simplex_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["simplex_x"] = 0.5 * result["stock_weight"] + result["t_bill_weight"]
    result["simplex_y"] = (math.sqrt(3) / 2) * result["stock_weight"]
    return result


def draw_simplex_outline(ax) -> None:
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, math.sqrt(3) / 2],
            [0.0, 0.0],
        ]
    )
    ax.plot(vertices[:, 0], vertices[:, 1], color="black", linewidth=0.8)
    ax.text(0.0, -0.05, "100% Bonds", ha="center", va="top", fontsize=11)
    ax.text(1.0, -0.05, "100% T-Bills", ha="center", va="top", fontsize=11)
    ax.text(
        0.5,
        math.sqrt(3) / 2 + 0.04,
        "100% Stocks",
        ha="center",
        va="bottom",
        fontsize=11,
    )
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, math.sqrt(3) / 2 + 0.08)
    ax.set_aspect("equal")
    ax.axis("off")
