"""
Visualize column impact analysis results.
Creates bar charts comparing feature importance and permutation importance.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(json_path: str = "output/column_impact.json") -> dict:
    """Load analysis results."""
    with open(json_path, "r") as f:
        return json.load(f)


def plot_comparison(results: dict, output_dir: Path) -> None:
    """Create comparison plot of both importance methods."""
    
    # Extract data
    feat_imp = {item["column"]: item["importance"] 
                for item in results["feature_importance"]["columns"]}
    perm_imp = {item["column"]: item["mean_auc_drop"] 
                for item in results["permutation_importance"]["columns"]}
    
    # Get common columns sorted by permutation importance
    columns = [item["column"] for item in results["permutation_importance"]["columns"]]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Feature Importance
    feat_values = [feat_imp[col] for col in columns]
    ax1.barh(range(len(columns)), feat_values, color='steelblue')
    ax1.set_yticks(range(len(columns)))
    ax1.set_yticklabels(columns)
    ax1.set_xlabel('Aggregated Feature Importance')
    ax1.set_title('CatBoost Feature Importance\n(by column)')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Permutation Importance
    perm_values = [perm_imp[col] for col in columns]
    ax2.barh(range(len(columns)), perm_values, color='coral')
    ax2.set_yticks(range(len(columns)))
    ax2.set_yticklabels(columns)
    ax2.set_xlabel('AUC Drop (mean)')
    ax2.set_title('Permutation Importance\n(performance impact when shuffled)')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "column_impact_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {output_file}")


def main() -> None:
    """Main execution."""
    results = load_results()
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_comparison(results, output_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TOP 5 MOST IMPORTANT COLUMNS")
    print("=" * 70)
    
    print("\nBy Feature Importance:")
    for item in results["feature_importance"]["columns"][:5]:
        print(f"  {item['column']:20s}: {item['importance']:8.2f}")
    
    print("\nBy Permutation Importance (AUC drop):")
    for item in results["permutation_importance"]["columns"][:5]:
        print(f"  {item['column']:20s}: {item['mean_auc_drop']:8.4f} ± {item['std_auc_drop']:.4f}")


if __name__ == "__main__":
    main()
