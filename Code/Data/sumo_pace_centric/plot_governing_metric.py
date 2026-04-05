import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import os

def main():
    plt.style.use('ggplot')

    # --- Scenario Data ---
    # Path A: Shorter distance but goes through city center (A* prefers this)
    # Path B: Longer distance but consistent highway speeds (MC prefers this)

    # 1. A* Cost Metrics (Purely Deterministic Time based on Distance/SpeedLimit)
    astar_cost_A = 100  # seconds
    astar_cost_B = 130  # seconds

    # 2. MC Cost Metrics (Stochastic Probability Distributions of Travel Time)
    # Path A (High Variance due to traffic lights, rush hour)
    mean_A = 160
    std_A = 80
    
    # Path B (Low Variance, very predictable highway)
    mean_B = 140
    std_B = 15

    # Lognormal parameters calculation
    def get_lognorm_params(mean, std):
        sigma2 = np.log(1 + (std/mean)**2)
        mu = np.log(mean) - sigma2/2
        sigma = np.sqrt(sigma2)
        return mu, sigma

    mu_A, sigma_A = get_lognorm_params(mean_A, std_A)
    mu_B, sigma_B = get_lognorm_params(mean_B, std_B)

    # Generate X-axis range (Travel Times)
    x = np.linspace(40, 450, 1000)
    
    # PDFs
    pdf_A = lognorm.pdf(x, sigma_A, scale=np.exp(mu_A))
    pdf_B = lognorm.pdf(x, sigma_B, scale=np.exp(mu_B))

    # CDFs
    cdf_A = lognorm.cdf(x, sigma_A, scale=np.exp(mu_A))
    cdf_B = lognorm.cdf(x, sigma_B, scale=np.exp(mu_B))

    deadline = 180  # Target arrival time T_d

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("The Governing Metric: Why Monte Carlo Makes Different Choices than A*", fontsize=18, fontweight='bold', y=0.98)

    # Subplot 1: Probability Density Functions (The Input Space)
    ax1.plot(x, pdf_A, color='#e63946', label='Path A (High Risk)', linewidth=2.5)
    ax1.plot(x, pdf_B, color='#457b9d', label='Path B (Reliable)', linewidth=2.5)
    
    # A* Perspectives
    ax1.axvline(astar_cost_A, color='#e63946', linestyle='--', linewidth=2, alpha=0.8, 
                label=f"A* Metric A (Deterministic {astar_cost_A}s)")
    ax1.axvline(astar_cost_B, color='#457b9d', linestyle='--', linewidth=2, alpha=0.8, 
                label=f"A* Metric B (Deterministic {astar_cost_B}s)")
    
    ax1.axvline(deadline, color='#2a9d8f', linestyle=':', linewidth=3, label=f"Deadline T_d ({deadline}s)")

    # Fill area under curve for MC visualization
    ax1.fill_between(x, pdf_A, where=(x < deadline), color='#e63946', alpha=0.15)
    ax1.fill_between(x, pdf_B, where=(x < deadline), color='#457b9d', alpha=0.15)

    ax1.set_title("1. How the Algorithms 'See' the Routes", fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel("Predicted Travel Time (Seconds)", fontsize=12)
    ax1.set_ylabel("Probability Density", fontsize=12)
    ax1.legend(fontsize=10, loc='upper right')
    
    ax1.text(0.05, 0.95, "A* chooses Path A strictly because 100s < 130s,\nentirely ignoring the massive fat tail of delay risk.", 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

    # Subplot 2: Cumulative Distribution Functions (The MC Optimization Goal)
    ax2.plot(x, cdf_A, color='#e63946', label='Path A Reliability', linewidth=2.5)
    ax2.plot(x, cdf_B, color='#457b9d', label='Path B Reliability', linewidth=2.5)
    ax2.axvline(deadline, color='#2a9d8f', linestyle=':', linewidth=3, label="Decision Boundary (Deadline)")

    # Calculate exact probabilities at the deadline
    prob_A = lognorm.cdf(deadline, sigma_A, scale=np.exp(mu_A))
    prob_B = lognorm.cdf(deadline, sigma_B, scale=np.exp(mu_B))

    ax2.plot(deadline, prob_A, marker='o', color='#e63946', markersize=10, markeredgecolor='black')
    ax2.plot(deadline, prob_B, marker='o', color='#457b9d', markersize=10, markeredgecolor='black')

    ax2.annotate(f"P(T < {deadline}s)\n= {prob_A:.1f}%", xy=(deadline, prob_A), 
                 xytext=(deadline + 15, prob_A - 0.1), color='#e63946', fontweight='bold', fontsize=12)
                 
    ax2.annotate(f"P(T < {deadline}s)\n= {prob_B:.1f}%", xy=(deadline, prob_B), 
                 xytext=(deadline - 90, prob_B + 0.05), color='#457b9d', fontweight='bold', fontsize=12)

    ax2.set_title("2. Monte Carlo Optimization Metric: Maximize Area $P(T < T_d)$", fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel("Predicted Travel Time (Seconds)", fontsize=12)
    ax2.set_ylabel("Cumulative Probability (Assurance)", fontsize=12)
    ax2.legend(fontsize=11)
    
    ax2.text(0.55, 0.15, "MC chooses Path B because its probability\nof arriving before the deadline is 99.4%,\ncompared to Path A's risky 66.8%.", 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_img = 'governing_metric_visualization.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"\n[+] Success! Plotted the governing mathematical inputs and saved to '{output_img}'")

if __name__ == "__main__":
    main()
