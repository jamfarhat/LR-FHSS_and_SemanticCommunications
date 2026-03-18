"""
LR-FHSS Simulation: Multi-Configuration Comparison
Compares different coding rates and header configurations
Format: [[success_ratio], [goodput_bytes], [transmitted], [AoI_media]]
"""

from lrfhss.run import run_sim
from lrfhss.settings import Settings
import time
from joblib import Parallel, delayed
import numpy as np
import matplotlib
# Use non-interactive backend to avoid display issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

N_NODES_POINTS = 15
N_NODES_MIN = 1000
N_NODES_MAX = 150000
LOOPS = 5
N_JOBS = 4

# Energy Efficiency Parameters
Pt = 0.1  # Default transmission power (watts)
LAMBDA = 1/900  # Traffic intensity: 1 packet every 900 seconds (default from Settings)

# Multiple test cases for comparison
TEST_CASES = [
    ('1/3', 3, 'DR8 - 1/3 rate, 3 headers'),
    ('2/3', 2, 'DR9 - 2/3 rate, 2 headers'),
]

N_NODES = np.linspace(N_NODES_MIN, N_NODES_MAX, N_NODES_POINTS, dtype=int) // 8

print("=" * 80)
print("LR-FHSS Simulation: Multi-Configuration Comparison")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print("Configurations to compare:")
for code_rate, headers, label in TEST_CASES:
    print(f"  • {label}")
print("=" * 80)
print()

# ============================================================================
# RUN SIMULATIONS
# ============================================================================

start_time = time.perf_counter()

# Storage for all configurations
all_results = {
    'nodes_devices': N_NODES * 8,
    'nodes_simulated': N_NODES,
}

nodes_devices = N_NODES * 8

for case_idx, (code_rate, headers, label) in enumerate(TEST_CASES):
    print()
    print(f"{'='*80}")
    print(f"Configuration {case_idx + 1}/{len(TEST_CASES)}: {label}")
    print(f"{'='*80}")
    
    success_results = []
    goodput_results = []
    aoi_results = []
    
    for idx, num_nodes in enumerate(N_NODES):
        print(f"[{idx + 1}/{len(N_NODES)}] Nodes: {num_nodes * 8} (simulating {num_nodes})...", 
              end=' ', flush=True)
        
        try:
            settings = Settings(number_nodes=num_nodes, code=code_rate, headers=headers)
            
            results = Parallel(n_jobs=N_JOBS)(
                delayed(run_sim)(settings, seed=seed) for seed in range(LOOPS)
            )
            
            success_list = []
            goodput_list = []
            aoi_list = []
            
            for r in results:
                try:
                    s = r[0][0] if isinstance(r[0], (list, np.ndarray)) else r[0]
                    g = r[1][0] if isinstance(r[1], (list, np.ndarray)) else r[1]
                    a = r[3][0] if isinstance(r[3], (list, np.ndarray)) else r[3]
                    
                    success_list.append(float(s))
                    goodput_list.append(float(g))
                    aoi_list.append(float(a))
                except (IndexError, TypeError, ValueError):
                    continue
            
            if success_list and goodput_list and aoi_list:
                success = np.mean(success_list)
                goodput = np.mean(goodput_list)
                aoi = np.mean(aoi_list)
            else:
                raise ValueError("Could not extract valid metrics")
            
            success_results.append(success)
            goodput_results.append(goodput)
            aoi_results.append(aoi)
            
            print(f"✓ Goodput: {goodput:.2f} bytes, AoI: {aoi:.4f}")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            success_results.append(0)
            goodput_results.append(0)
            aoi_results.append(0)
    
    # Store results for this configuration
    all_results[f'{label}_success'] = success_results
    all_results[f'{label}_goodput'] = goodput_results
    all_results[f'{label}_aoi'] = aoi_results
    
    # Calculate efficiency for each node count
    efficiency_results = []
    for idx, num_nodes in enumerate(N_NODES):
        code_rate = TEST_CASES[case_idx][0]
        headers = TEST_CASES[case_idx][1]
        settings_temp = Settings(number_nodes=num_nodes, code=code_rate, headers=headers)
        
        goodput = goodput_results[idx]
        header_duration = settings_temp.header_duration
        payload_duration = settings_temp.payload_duration
        num_headers = settings_temp.headers
        num_payloads = settings_temp.payloads
        
        energy_denominator = Pt * (num_nodes * (1/LAMBDA) * (num_headers * header_duration + num_payloads * payload_duration))
        efficiency = goodput / energy_denominator if energy_denominator > 0 else 0
        efficiency_results.append(efficiency)
    
    all_results[f'{label}_efficiency'] = efficiency_results

elapsed_time = time.perf_counter() - start_time

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

print()
print("=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print(f"Simulation time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print()

for case_idx, (code_rate, headers, label) in enumerate(TEST_CASES):
    print(f"\n{label}")
    print("-" * 100)
    print("Nodes (total) | Goodput (bytes) | AoI          | Success Ratio | Energy Efficiency")
    print("-" * 100)
    
    for i, nodes in enumerate(nodes_devices):
        goodput = all_results[f'{label}_goodput'][i]
        aoi = all_results[f'{label}_aoi'][i]
        success = all_results[f'{label}_success'][i]
        efficiency = all_results[f'{label}_efficiency'][i]
        print(f"{nodes:>12} | {goodput:>15.2f} | {aoi:>12.4f} | {success:>13.4f} | {efficiency:>16.6f}")

print()

# ============================================================================
# SAVE RESULTS
# ============================================================================

output_dir = 'simulation_results'
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save results for each configuration
for case_idx, (code_rate, headers, label) in enumerate(TEST_CASES):
    results_df = pd.DataFrame({
        'Nodes_Total': nodes_devices,
        'Nodes_Simulated': N_NODES,
        'Success_Ratio': all_results[f'{label}_success'],
        'Goodput_bytes': all_results[f'{label}_goodput'],
        'AoI': all_results[f'{label}_aoi'],
        'Energy_Efficiency': all_results[f'{label}_efficiency']
    })
    
    # Sanitize filename: remove special characters
    safe_label = label.replace(" ", "_").replace(",", "").replace("/", "_").replace("-", "").replace(".", "")
    csv_file = f'{output_dir}/results_case{case_idx + 1}_{timestamp}.csv'
    results_df.to_csv(csv_file, index=False)
    print(f"✓ Results saved: {csv_file}")

print()

# ============================================================================
# GENERATE COMBINED COMPARISON PLOT
# ============================================================================

print("Generating combined comparison plot...")

fig_combined = plt.figure(figsize=(24, 5))

# Extract data
label_dr8 = TEST_CASES[0][2]
label_dr9 = TEST_CASES[1][2]
goodput_dr8 = all_results[f'{label_dr8}_goodput']
aoi_dr8 = all_results[f'{label_dr8}_aoi']
efficiency_dr8 = all_results[f'{label_dr8}_efficiency']

goodput_dr9 = all_results[f'{label_dr9}_goodput']
aoi_dr9 = all_results[f'{label_dr9}_aoi']
efficiency_dr9 = all_results[f'{label_dr9}_efficiency']

# Plot 1: Goodput - Bar Chart Comparison
ax1 = plt.subplot(1, 5, 1)
x = np.arange(len(nodes_devices))
width = 0.35
bars1 = ax1.bar(x - width/2, goodput_dr8, width, label=label_dr8, color='#1f77b4', alpha=0.8)
bars2 = ax1.bar(x + width/2, goodput_dr9, width, label=label_dr9, color='#ff7f0e', alpha=0.8)
ax1.set_xlabel("Device Count Index", fontsize=12, fontweight='bold')
ax1.set_ylabel("Goodput (bytes)", fontsize=12, fontweight='bold')
ax1.set_title("Goodput Comparison", fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f'{d//1000}K' for d in nodes_devices], rotation=45, ha='right', fontsize=9)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')

# Plot 2: AoI - Bar Chart Comparison
ax2 = plt.subplot(1, 5, 2)
bars1 = ax2.bar(x - width/2, aoi_dr8, width, label=label_dr8, color='#1f77b4', alpha=0.8)
bars2 = ax2.bar(x + width/2, aoi_dr9, width, label=label_dr9, color='#ff7f0e', alpha=0.8)
ax2.set_xlabel("Device Count Index", fontsize=12, fontweight='bold')
ax2.set_ylabel("AoI", fontsize=12, fontweight='bold')
ax2.set_title("AoI Comparison", fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f'{d//1000}K' for d in nodes_devices], rotation=45, ha='right', fontsize=9)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

# Plot 3: Energy Efficiency - Bar Chart Comparison
ax3 = plt.subplot(1, 5, 3)
bars1 = ax3.bar(x - width/2, efficiency_dr8, width, label=label_dr8, color='#1f77b4', alpha=0.8)
bars2 = ax3.bar(x + width/2, efficiency_dr9, width, label=label_dr9, color='#ff7f0e', alpha=0.8)
ax3.set_xlabel("Device Count Index", fontsize=12, fontweight='bold')
ax3.set_ylabel("Energy Efficiency (bytes/J)", fontsize=12, fontweight='bold')
ax3.set_title("Energy Efficiency Comparison", fontsize=13, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels([f'{d//1000}K' for d in nodes_devices], rotation=45, ha='right', fontsize=9)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y', linestyle='--')

# Plot 4: Goodput vs AoI Trade-off
ax4 = plt.subplot(1, 5, 4)
ax4.scatter(aoi_dr8, goodput_dr8, s=250, marker='o', color='#1f77b4', 
           edgecolors='black', linewidth=2, alpha=0.8, label=label_dr8, zorder=3)
ax4.plot(aoi_dr8, goodput_dr8, 'b--', alpha=0.4, linewidth=2)
ax4.scatter(aoi_dr9, goodput_dr9, s=250, marker='s', color='#ff7f0e', 
           edgecolors='black', linewidth=2, alpha=0.8, label=label_dr9, zorder=3)
ax4.plot(aoi_dr9, goodput_dr9, color='#ff7f0e', linestyle='--', alpha=0.4, linewidth=2)
ax4.set_xlabel("Average Age of Information (AoI)", fontsize=12, fontweight='bold')
ax4.set_ylabel("Goodput (bytes)", fontsize=12, fontweight='bold')
ax4.set_title("Goodput vs AoI Trade-off", fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.legend(fontsize=11, loc='best')

# Plot 5: AoI vs Energy Efficiency Trade-off
ax5 = plt.subplot(1, 5, 5)
ax5.scatter(aoi_dr8, efficiency_dr8, s=250, marker='o', color='#1f77b4', 
           edgecolors='black', linewidth=2, alpha=0.8, label=label_dr8, zorder=3)
ax5.plot(aoi_dr8, efficiency_dr8, 'b--', alpha=0.4, linewidth=2)
ax5.scatter(aoi_dr9, efficiency_dr9, s=250, marker='s', color='#ff7f0e', 
           edgecolors='black', linewidth=2, alpha=0.8, label=label_dr9, zorder=3)
ax5.plot(aoi_dr9, efficiency_dr9, color='#ff7f0e', linestyle='--', alpha=0.4, linewidth=2)
ax5.set_xlabel("Average Age of Information (AoI)", fontsize=12, fontweight='bold')
ax5.set_ylabel("Energy Efficiency (bytes/J)", fontsize=12, fontweight='bold')
ax5.set_title("AoI vs Energy Efficiency Trade-off", fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3, linestyle='--')
ax5.legend(fontsize=11, loc='best')

plt.tight_layout()

combined_plot_file = f'{output_dir}/combined_comparison_{timestamp}.png'
fig_combined.savefig(combined_plot_file, dpi=150, bbox_inches='tight')
print(f"✓ Combined plot saved to: {combined_plot_file}")
print()

print("=" * 80)
print("✓ Simulation complete!")
print("=" * 80)
print(f"\nResults location: {os.path.abspath(output_dir)}")
print("\nGenerated files:")
for file in sorted(os.listdir(output_dir)):
    if timestamp in file:
        print(f"  • {file}")


