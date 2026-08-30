import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    trace_path = Path("artifacts/evaluation/mastery-trace.csv")
    output_path = Path("artifacts/evaluation/plots/true_vs_tracked_knowledge.png")
    
    # Read trace
    df = pd.read_csv(trace_path)
    
    # Policies in order
    policies = ["static", "legacy_rule", "improved_rule", "ml_based"]
    
    # Group by policy and step, average true_knowledge and tracked_K
    grouped = df.groupby(['policy', 'step'])[['true_knowledge', 'tracked_K']].mean().reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    axes = axes.flatten()
    
    for i, policy in enumerate(policies):
        ax = axes[i]
        policy_df = grouped[grouped['policy'] == policy]
        
        # Sort by step just in case
        policy_df = policy_df.sort_values('step')
        
        # Plot
        ax.plot(policy_df['step'], policy_df['true_knowledge'], color='blue', label='Simulator True K')
        ax.plot(policy_df['step'], policy_df['tracked_K'], color='red', linestyle='--', label='Rule Engine Tracked K')
        
        ax.set_title(policy)
        ax.set_xlabel("Step")
        ax.set_ylabel("Knowledge")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 150)
        ax.set_ylim(0, 1)
        
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
