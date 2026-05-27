"""
Visualize comprehensive column impact analysis.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(json_path: str = "output/all_columns_impact.json") -> dict:
    """Load analysis results."""
    with open(json_path, "r") as f:
        return json.load(f)


def plot_column_impact(results: dict, output_dir: Path) -> None:
    """Create bar chart of all column impacts."""
    
    columns = list(results["columns"].keys())
    impacts = [results["columns"][col]["auc_drop"] for col in columns]
    methods = [results["columns"][col]["method"] for col in columns]
    
    # Color by method
    colors = ['coral' if m == 'ablation' else 'steelblue' for m in methods]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.barh(range(len(columns)), impacts, color=colors)
    ax.set_yticks(range(len(columns)))
    ax.set_yticklabels(columns)
    ax.set_xlabel('AUC Drop (Higher = More Important)')
    ax.set_title(f'Input Column Impact on Predictions\nBaseline AUC: {results["baseline_auc"]:.4f}')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='coral', label='Text (Ablation)'),
        Patch(facecolor='steelblue', label='Numeric/Categorical (Permutation)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    output_file = output_dir / "all_columns_impact.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {output_file}")


def main() -> None:
    """Main execution."""
    results = load_results()
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_column_impact(results, output_dir)
    
    # Print top columns
    print("\n" + "=" * 70)
    print("TOP 5 MOST IMPORTANT INPUT COLUMNS")
    print("=" * 70)
    
    for i, (col, data) in enumerate(list(results["columns"].items())[:5], 1):
        method = data["method"].capitalize()
        std_str = f" ± {data['std']:.4f}" if "std" in data else ""
        print(f"{i}. {col:20s} ({method:12s}): {data['auc_drop']:+.4f}{std_str} ({data['percentage_impact']:+.2f}%)")


if __name__ == "__main__":
    main()
