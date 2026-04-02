#!/usr/bin/env python3
"""
Comprehensive Metrics Comparison: DR8 vs DR9 vs Semantic Communication

Compares four key performance metrics:
1. Success Rate (%)
2. Goodput (bits/second)
3. Energy Efficiency (bits/Joule)
4. Age of Information (AoI) (seconds)

Across three protocols:
- DR8 (traditional LoRa - Exponential Traffic)
- DR9 (traditional LoRa - Exponential Traffic)
- Semantic (proposed semantic communication - SemanticTraffic)

Output: Publication-ready visualization with 4 comparison plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lrfhss.settings import Settings
from lrfhss.run import run_sim
from lrfhss.traffic import Exponential_Traffic, SemanticTraffic


class MetricsComparison:
    """Comprehensive metrics comparison across protocols and network sizes"""
    
    def __init__(self):
        self.results = {
            'dr8': {'success': [], 'goodput': [], 'energy': [], 'aoi': []},
            'dr9': {'success': [], 'goodput': [], 'energy': [], 'aoi': []},
            'semantic': {'success': [], 'goodput': [], 'energy': [], 'aoi': []}
        }
        self.node_counts = [10000, 30000, 50000, 70000, 90000, 110000, 130000, 150000, 170000, 190000]
        
    def calculate_metrics(self, results_list, num_nodes, protocol_name):
        """
        Calculate all metrics from simulation results
        
        Args:
            results_list: Simulation results from run_sim()
                         Format: [[success_rate], [goodput], [transmitted], [AoI_media]]
            num_nodes: Number of nodes in network
            protocol_name: Name of protocol (dr8, dr9, semantic)
        """
        metrics = {}
        
        # Handle case where simulation returns 1 (no transmissions)
        if results_list == 1:
            metrics['success_rate'] = 100.0
            metrics['goodput'] = 0.0
            metrics['energy_efficiency'] = 0.0
            metrics['aoi'] = 0.0
            return metrics
        
        # Extract results from list format
        # [[success_rate], [goodput], [transmitted], [AoI_media]]
        success_rate_ratio = results_list[0][0]  # Already a ratio (0-1)
        goodput_bits = results_list[1][0]  # Total successful bits
        total_transmissions = results_list[2][0]  # Total transmissions
        aoi_media = results_list[3][0]  # Average Age of Information
        
        # ==================== SUCCESS RATE ====================
        # Convert ratio to percentage
        success_rate = success_rate_ratio * 100
        metrics['success_rate'] = success_rate
        
        # ==================== GOODPUT ====================
        # Goodput = (successful bits) / simulation_time
        simulation_time = 3600  # seconds (default from Settings)
        goodput_bits_per_sec = goodput_bits / simulation_time
        metrics['goodput'] = goodput_bits_per_sec
        
        # ==================== ENERGY EFFICIENCY ====================
        # Energy per transmission: 0.1 Joules (typical LoRa module)
        energy_per_tx = 0.1  # Joules
        total_energy = total_transmissions * energy_per_tx  # Joules
        
        if total_energy > 0:
            energy_efficiency = goodput_bits / total_energy  # bits/Joule
        else:
            energy_efficiency = 0.0
        metrics['energy_efficiency'] = energy_efficiency
        
        # ==================== AGE OF INFORMATION (AoI) ====================
        # AoI is already calculated in simulation
        metrics['aoi'] = aoi_media
        
        return metrics
    
    def run_dr8_simulation(self, num_nodes):
        """Run DR8 (Traditional LoRa with Exponential Traffic)"""
        print(f"  → Running DR8 simulation ({num_nodes:,} nodes)...")
        
        settings = Settings(
            number_nodes=num_nodes//8,
            traffic_class=Exponential_Traffic,
            traffic_param={'average_interval': 600}, 
            code='1/3', 
            headers=3
        )
        
        try:
            results = run_sim(settings)
            metrics = self.calculate_metrics(results, num_nodes, 'DR8')
            return metrics
        except Exception as e:
            print(f"     ERROR: {e}")
            return None
    
    def run_dr9_simulation(self, num_nodes):
        """Run DR9 (Traditional LoRa with Exponential Traffic, different interval)"""
        print(f"  → Running DR9 simulation ({num_nodes:,} nodes)...")
        
        settings = Settings(
            number_nodes=num_nodes//8,
            traffic_class=Exponential_Traffic,
            traffic_param={'average_interval': 600},
            code='2/3', 
            headers=2
        )
        
        try:
            results = run_sim(settings)
            metrics = self.calculate_metrics(results, num_nodes, 'DR9')
            return metrics
        except Exception as e:
            print(f"     ERROR: {e}")
            return None
    
    def run_semantic_simulation(self, num_nodes):
        """Run Semantic Communication simulation"""
        print(f"  → Running Semantic simulation ({num_nodes:,} nodes)...")
        
        settings = Settings(
            number_nodes=num_nodes//8,
            traffic_class=SemanticTraffic,
            traffic_param={
                'average_interval': 600,  # Same base interval
                'threshold_0': 0.75,  # Base semantic threshold
                'beta': 0.00005  # Adaptive parameter
            }
        )
        
        try:
            results = run_sim(settings)
            metrics = self.calculate_metrics(results, num_nodes, 'Semantic')
            return metrics
        except Exception as e:
            print(f"     ERROR: {e}")
            return None
    
    def run_all_simulations(self):
        """Run all simulations for all network sizes"""
        print("\n" + "="*70)
        print("COMPREHENSIVE METRICS COMPARISON: DR8 vs DR9 vs SEMANTIC")
        print("="*70)
        
        for i, num_nodes in enumerate(self.node_counts, 1):
            print(f"\n[{i}/{len(self.node_counts)}] Network Size: {num_nodes:,} nodes")
            print("-" * 70)
            
            # Run DR8
            dr8_metrics = self.run_dr8_simulation(num_nodes)
            if dr8_metrics:
                self.results['dr8']['success'].append(dr8_metrics['success_rate'])
                self.results['dr8']['goodput'].append(dr8_metrics['goodput'])
                self.results['dr8']['energy'].append(dr8_metrics['energy_efficiency'])
                self.results['dr8']['aoi'].append(dr8_metrics['aoi'])
                print(f"     Success: {dr8_metrics['success_rate']:.1f}% | "
                      f"Goodput: {dr8_metrics['goodput']:.2f} bps | "
                      f"Energy: {dr8_metrics['energy_efficiency']:.2f} bits/J | "
                      f"AoI: {dr8_metrics['aoi']:.1f}s")
            
            # Run DR9
            dr9_metrics = self.run_dr9_simulation(num_nodes)
            if dr9_metrics:
                self.results['dr9']['success'].append(dr9_metrics['success_rate'])
                self.results['dr9']['goodput'].append(dr9_metrics['goodput'])
                self.results['dr9']['energy'].append(dr9_metrics['energy_efficiency'])
                self.results['dr9']['aoi'].append(dr9_metrics['aoi'])
                print(f"     Success: {dr9_metrics['success_rate']:.1f}% | "
                      f"Goodput: {dr9_metrics['goodput']:.2f} bps | "
                      f"Energy: {dr9_metrics['energy_efficiency']:.2f} bits/J | "
                      f"AoI: {dr9_metrics['aoi']:.1f}s")
            
            # Run Semantic
            semantic_metrics = self.run_semantic_simulation(num_nodes)
            if semantic_metrics:
                self.results['semantic']['success'].append(semantic_metrics['success_rate'])
                self.results['semantic']['goodput'].append(semantic_metrics['goodput'])
                self.results['semantic']['energy'].append(semantic_metrics['energy_efficiency'])
                self.results['semantic']['aoi'].append(semantic_metrics['aoi'])
                print(f"     Success: {semantic_metrics['success_rate']:.1f}% | "
                      f"Goodput: {semantic_metrics['goodput']:.2f} bps | "
                      f"Energy: {semantic_metrics['energy_efficiency']:.2f} bits/J | "
                      f"AoI: {semantic_metrics['aoi']:.1f}s")
    
    def create_visualization(self):
        """Create publication-ready visualization"""
        print("\n" + "="*70)
        print("GENERATING VISUALIZATION")
        print("="*70)
        
        # Prepare data
        x_pos = np.arange(len(self.node_counts))
        x_labels = [f"{n//1000}K" for n in self.node_counts]
        
        # Create figure with 2x2 grid for 4 metrics
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        colors = {'dr8': '#FF6B6B', 'dr9': '#FFA500', 'semantic': '#4ECDC4'}
        
        # ==================== SUCCESS RATE ====================
        ax1 = fig.add_subplot(gs[0, 0])
        width = 0.25
        
        ax1.bar(x_pos - width, self.results['dr8']['success'], width, 
                label='DR8', color=colors['dr8'], alpha=0.8)
        ax1.bar(x_pos, self.results['dr9']['success'], width, 
                label='DR9', color=colors['dr9'], alpha=0.8)
        ax1.bar(x_pos + width, self.results['semantic']['success'], width, 
                label='Semantic', color=colors['semantic'], alpha=0.8)
        
        ax1.set_xlabel('Network Size', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
        ax1.set_title('Success Rate Comparison', fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(x_labels)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 105)
        
        # Add value labels on bars
        for i, v in enumerate(self.results['dr8']['success']):
            if v > 0:
                ax1.text(i - width, v + 2, f'{v:.0f}%', ha='center', fontsize=8)
        for i, v in enumerate(self.results['dr9']['success']):
            if v > 0:
                ax1.text(i, v + 2, f'{v:.0f}%', ha='center', fontsize=8)
        for i, v in enumerate(self.results['semantic']['success']):
            if v > 0:
                ax1.text(i + width, v + 2, f'{v:.0f}%', ha='center', fontsize=8)
        
        # ==================== GOODPUT ====================
        ax2 = fig.add_subplot(gs[0, 1])
        
        ax2.plot(x_pos, self.results['dr8']['goodput'], 'o-', linewidth=2, 
                markersize=8, label='DR8', color=colors['dr8'], alpha=0.8)
        ax2.plot(x_pos, self.results['dr9']['goodput'], 's-', linewidth=2, 
                markersize=8, label='DR9', color=colors['dr9'], alpha=0.8)
        ax2.plot(x_pos, self.results['semantic']['goodput'], '^-', linewidth=2, 
                markersize=8, label='Semantic', color=colors['semantic'], alpha=0.8)
        
        ax2.set_xlabel('Network Size', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Goodput (bits/second)', fontsize=11, fontweight='bold')
        ax2.set_title('Goodput Comparison', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(x_labels)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # ==================== ENERGY EFFICIENCY ====================
        ax3 = fig.add_subplot(gs[1, 0])
        
        ax3.plot(x_pos, self.results['dr8']['energy'], 'o-', linewidth=2, 
                markersize=8, label='DR8', color=colors['dr8'], alpha=0.8)
        ax3.plot(x_pos, self.results['dr9']['energy'], 's-', linewidth=2, 
                markersize=8, label='DR9', color=colors['dr9'], alpha=0.8)
        ax3.plot(x_pos, self.results['semantic']['energy'], '^-', linewidth=2, 
                markersize=8, label='Semantic', color=colors['semantic'], alpha=0.8)
        
        ax3.set_xlabel('Network Size', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Energy Efficiency (bits/Joule)', fontsize=11, fontweight='bold')
        ax3.set_title('Energy Efficiency Comparison', fontsize=12, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(x_labels)
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # ==================== AGE OF INFORMATION ====================
        ax4 = fig.add_subplot(gs[1, 1])
        
        ax4.plot(x_pos, self.results['dr8']['aoi'], 'o-', linewidth=2, 
                markersize=8, label='DR8', color=colors['dr8'], alpha=0.8)
        ax4.plot(x_pos, self.results['dr9']['aoi'], 's-', linewidth=2, 
                markersize=8, label='DR9', color=colors['dr9'], alpha=0.8)
        ax4.plot(x_pos, self.results['semantic']['aoi'], '^-', linewidth=2, 
                markersize=8, label='Semantic', color=colors['semantic'], alpha=0.8)
        
        ax4.set_xlabel('Network Size', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Age of Information (seconds)', fontsize=11, fontweight='bold')
        ax4.set_title('AoI Comparison (Lower is Better)', fontsize=12, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(x_labels)
        ax4.legend(loc='best', fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # Overall title
        fig.suptitle('Comprehensive Metrics Comparison: DR8 vs DR9 vs Semantic Communication',
                     fontsize=14, fontweight='bold', y=0.995)
        
        # Save figure
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  'simulation_results')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'metrics_comparison_dr8_dr9_semantic.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: {output_path}")
        
        plt.close()
    
    def print_summary_table(self):
        """Print summary statistics table"""
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        
        print("\nSUCCESS RATE (%)")
        print("-" * 70)
        print(f"{'Network':<15} {'DR8':<20} {'DR9':<20} {'Semantic':<15}")
        print("-" * 70)
        for i, size in enumerate(self.node_counts):
            if i < len(self.results['dr8']['success']):
                print(f"{size//1000}K nodes{'':<7} {self.results['dr8']['success'][i]:>6.1f}% "
                      f"{'':>13} {self.results['dr9']['success'][i]:>6.1f}% "
                      f"{'':>13} {self.results['semantic']['success'][i]:>6.1f}%")
        
        print("\nGOODPUT (bits/second)")
        print("-" * 70)
        print(f"{'Network':<15} {'DR8':<20} {'DR9':<20} {'Semantic':<15}")
        print("-" * 70)
        for i, size in enumerate(self.node_counts):
            if i < len(self.results['dr8']['goodput']):
                print(f"{size//1000}K nodes{'':<7} {self.results['dr8']['goodput'][i]:>10.2f} "
                      f"{'':>9} {self.results['dr9']['goodput'][i]:>10.2f} "
                      f"{'':>9} {self.results['semantic']['goodput'][i]:>10.2f}")
        
        print("\nENERGY EFFICIENCY (bits/Joule)")
        print("-" * 70)
        print(f"{'Network':<15} {'DR8':<20} {'DR9':<20} {'Semantic':<15}")
        print("-" * 70)
        for i, size in enumerate(self.node_counts):
            if i < len(self.results['dr8']['energy']):
                print(f"{size//1000}K nodes{'':<7} {self.results['dr8']['energy'][i]:>10.2f} "
                      f"{'':>9} {self.results['dr9']['energy'][i]:>10.2f} "
                      f"{'':>9} {self.results['semantic']['energy'][i]:>10.2f}")
        
        print("\nAGE OF INFORMATION (seconds) - Lower is Better")
        print("-" * 70)
        print(f"{'Network':<15} {'DR8':<20} {'DR9':<20} {'Semantic':<15}")
        print("-" * 70)
        for i, size in enumerate(self.node_counts):
            if i < len(self.results['dr8']['aoi']):
                print(f"{size//1000}K nodes{'':<7} {self.results['dr8']['aoi'][i]:>10.1f}s "
                      f"{'':>8} {self.results['dr9']['aoi'][i]:>10.1f}s "
                      f"{'':>8} {self.results['semantic']['aoi'][i]:>10.1f}s")
        
        # Calculate improvements
        print("\n" + "="*70)
        print("SEMANTIC IMPROVEMENT OVER TRADITIONAL PROTOCOLS")
        print("="*70)
        
        for i, size in enumerate(self.node_counts):
            if i < len(self.results['dr8']['success']):
                print(f"\n{size//1000}K nodes:")
                
                # vs DR8
                if self.results['dr8']['success'][i] > 0:
                    success_improvement_dr8 = ((self.results['semantic']['success'][i] - 
                                               self.results['dr8']['success'][i]) / 
                                               max(self.results['dr8']['success'][i], 1) * 100)
                else:
                    success_improvement_dr8 = 0
                    
                if self.results['dr8']['goodput'][i] > 0:
                    goodput_improvement_dr8 = ((self.results['semantic']['goodput'][i] - 
                                               self.results['dr8']['goodput'][i]) / 
                                               max(self.results['dr8']['goodput'][i], 1) * 100)
                else:
                    goodput_improvement_dr8 = 0
                    
                if self.results['dr8']['energy'][i] > 0:
                    energy_improvement_dr8 = ((self.results['semantic']['energy'][i] - 
                                              self.results['dr8']['energy'][i]) / 
                                              max(self.results['dr8']['energy'][i], 1) * 100)
                else:
                    energy_improvement_dr8 = 0
                    
                if self.results['dr8']['aoi'][i] > 0:
                    aoi_improvement_dr8 = ((self.results['dr8']['aoi'][i] - 
                                           self.results['semantic']['aoi'][i]) / 
                                           max(self.results['dr8']['aoi'][i], 1) * 100)
                else:
                    aoi_improvement_dr8 = 0
                
                print(f"  vs DR8:")
                print(f"    Success Rate:    {success_improvement_dr8:+.1f}%")
                print(f"    Goodput:         {goodput_improvement_dr8:+.1f}%")
                print(f"    Energy Eff.:     {energy_improvement_dr8:+.1f}%")
                print(f"    AoI (lower is better): {aoi_improvement_dr8:+.1f}%")
                
                # vs DR9
                if self.results['dr9']['success'][i] > 0:
                    success_improvement_dr9 = ((self.results['semantic']['success'][i] - 
                                               self.results['dr9']['success'][i]) / 
                                               max(self.results['dr9']['success'][i], 1) * 100)
                else:
                    success_improvement_dr9 = 0
                    
                if self.results['dr9']['goodput'][i] > 0:
                    goodput_improvement_dr9 = ((self.results['semantic']['goodput'][i] - 
                                               self.results['dr9']['goodput'][i]) / 
                                               max(self.results['dr9']['goodput'][i], 1) * 100)
                else:
                    goodput_improvement_dr9 = 0
                    
                if self.results['dr9']['energy'][i] > 0:
                    energy_improvement_dr9 = ((self.results['semantic']['energy'][i] - 
                                              self.results['dr9']['energy'][i]) / 
                                              max(self.results['dr9']['energy'][i], 1) * 100)
                else:
                    energy_improvement_dr9 = 0
                    
                if self.results['dr9']['aoi'][i] > 0:
                    aoi_improvement_dr9 = ((self.results['dr9']['aoi'][i] - 
                                           self.results['semantic']['aoi'][i]) / 
                                           max(self.results['dr9']['aoi'][i], 1) * 100)
                else:
                    aoi_improvement_dr9 = 0
                
                print(f"  vs DR9:")
                print(f"    Success Rate:    {success_improvement_dr9:+.1f}%")
                print(f"    Goodput:         {goodput_improvement_dr9:+.1f}%")
                print(f"    Energy Eff.:     {energy_improvement_dr9:+.1f}%")
                print(f"    AoI (lower is better): {aoi_improvement_dr9:+.1f}%")


def main():
    """Main execution"""
    comparison = MetricsComparison()
    
    try:
        # Run all simulations
        comparison.run_all_simulations()
        
        # Print summary
        comparison.print_summary_table()
        
        # Create visualization
        comparison.create_visualization()
        
        print("\n" + "="*70)
        print("✓ ANALYSIS COMPLETE")
        print("="*70)
        print("\nOutput files:")
        print("  • simulation_results/metrics_comparison_dr8_dr9_semantic.png")
        print("\nNext steps:")
        print("  1. Review the generated PNG visualization")
        print("  2. Use the summary statistics for your report")
        print("  3. Compare metrics with your specific requirements")
        
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Execution cancelled by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()