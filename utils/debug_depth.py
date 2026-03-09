from __future__ import annotations
import sys
sys.path.append("/home/rain/Desktop/workspace/APS360/riskmap")
from dataloader import compute_depth_weight_value
from train import DEPTH_MAX, DEPTH_MIN, debug_depth


# Edit these lists if you want to inspect different depth values.
DEPTH_VALUES = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0]
COMPARE_PAIRS = [
    (1.0, 2.0),
    (2.0, 5.0),
    (5.0, 10.0),
    (10.0, 15.0),
    (15.0, 20.0),
    (10.0, 20.0),
    (20.0, 40.0),
]


def compare_depth_pairs(depth_pairs: list[tuple[float, float]]) -> list[dict[str, float]]:
    """Compare how much the weight changes between two depth values."""
    results: list[dict[str, float]] = []

    print()
    print("Pairwise depth comparisons")
    print("depth_a\tdepth_b\tweight_a\tweight_b\tdelta_b_minus_a")

    for depth_a, depth_b in depth_pairs:
        weight_a = compute_depth_weight_value(
            depth_value=float(depth_a),
            depth_min=DEPTH_MIN,
            depth_max=DEPTH_MAX,
        )
        weight_b = compute_depth_weight_value(
            depth_value=float(depth_b),
            depth_min=DEPTH_MIN,
            depth_max=DEPTH_MAX,
        )
        delta = weight_b - weight_a

        print(
            f"{float(depth_a):.4f}\t{float(depth_b):.4f}\t"
            f"{weight_a:.6f}\t{weight_b:.6f}\t{delta:.6f}"
        )

        results.append(
            {
                "depth_a": float(depth_a),
                "depth_b": float(depth_b),
                "weight_a": float(weight_a),
                "weight_b": float(weight_b),
                "delta_b_minus_a": float(delta),
            }
        )

    return results


def print_single_depth(depth_value: float) -> None:
    weight = compute_depth_weight_value(
        depth_value=float(depth_value),
        depth_min=DEPTH_MIN,
        depth_max=DEPTH_MAX,
    )
    print()
    print(
        f"Single depth check: depth={float(depth_value):.4f} "
        f"-> weight={weight:.6f}"
    )


def main() -> None:
    print(
        "Depth weighting debug script\n"
        f"Using DEPTH_MIN={DEPTH_MIN:.4f}, DEPTH_MAX={DEPTH_MAX:.4f}"
    )
    debug_depth(DEPTH_VALUES)
    compare_depth_pairs(COMPARE_PAIRS)
    print_single_depth(10.0)


if __name__ == "__main__":
    main()
