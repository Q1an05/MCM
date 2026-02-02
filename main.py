#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C: Data With The Stars
Main Pipeline - Sequential Execution of All Analysis Scripts

This script executes all analysis modules in the correct dependency order:
1. Data Processing
2. Monte Carlo Simulations (Basic & Bayesian)
3. Model Evaluation
4. Rule Comparison & System Diagnostics
5. Factor Analysis
6. System Design & Case Studies

Author: MCM Team
Date: 2026-02-02
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import time

# Configuration
BASE_DIR = Path(__file__).parent
SRC_DIR = BASE_DIR / "src"

# Pipeline steps in execution order
PIPELINE_STEPS = [
    {
        'step': 1,
        'name': 'Data Processing',
        'script': 'data_processing.py',
        'description': 'Process raw data and create simulation input'
    },
    {
        'step': 2,
        'name': 'Basic Monte Carlo Simulation',
        'script': 'monte_carlo_sim.py',
        'description': 'Run basic Monte Carlo fan vote estimation'
    },
    {
        'step': 3,
        'name': 'Bayesian Monte Carlo Simulation',
        'script': 'monte_carlo_bayesian_dirichlet.py',
        'description': 'Run advanced Bayesian-Dirichlet Monte Carlo'
    },
    {
        'step': 4,
        'name': 'Model Evaluation',
        'script': 'evaluate_models.py',
        'description': 'Compare Basic vs Bayesian model performance'
    },
    {
        'step': 5,
        'name': 'Rule Comparison',
        'script': 'rule_comparison.py',
        'description': 'Counterfactual analysis of elimination rules'
    },
    {
        'step': 6,
        'name': 'System Diagnostics',
        'script': 'system_diagnostics.py',
        'description': 'Stress testing and decision boundary analysis'
    },
    {
        'step': 7,
        'name': 'Judges Save Analysis',
        'script': 'analyze_judges_save.py',
        'description': 'Analyze rationality of judges\' save decisions'
    },
    {
        'step': 8,
        'name': 'Factor Analysis',
        'script': 'factor_analysis_v2.py',
        'description': 'Advanced factor analysis with clustering'
    },
    {
        'step': 9,
        'name': 'System Design Optimization',
        'script': 'system_design.py',
        'description': 'Optimize DTPM system parameters'
    },
    {
        'step': 10,
        'name': 'Case Study Analysis',
        'script': 'case_study_analysis.py',
        'description': 'Analyze controversial contestant cases'
    }
]


def print_header():
    """Print pipeline header"""
    print("=" * 70)
    print(" MCM 2026 Problem C - Complete Analysis Pipeline")
    print("=" * 70)
    print(f" Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Total Steps: {len(PIPELINE_STEPS)}")
    print("=" * 70)
    print()


def print_step_header(step_info):
    """Print step header"""
    print("\n" + "─" * 70)
    print(f"Step {step_info['step']}/{len(PIPELINE_STEPS)}: {step_info['name']}")
    print("─" * 70)
    print(f"Description: {step_info['description']}")
    print(f"Script: {step_info['script']}")
    print("─" * 70)


def run_script(script_path: Path, step_info: dict) -> tuple:
    """
    Run a Python script and return (success, duration)
    
    Args:
        script_path: Path to the script
        step_info: Dictionary with step information
        
    Returns:
        Tuple of (success: bool, duration: float)
    """
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )
        
        duration = time.time() - start_time
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        # Check for errors
        if result.returncode != 0:
            print(f"\n[ERROR] Script failed with return code {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            return False, duration
        
        print(f"\n✓ Completed in {duration:.2f} seconds")
        return True, duration
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"\n[ERROR] Script timed out after {duration:.2f} seconds")
        return False, duration
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"\n[ERROR] Unexpected error: {str(e)}")
        return False, duration


def print_summary(results: list, total_duration: float):
    """Print pipeline summary"""
    print("\n" + "=" * 70)
    print(" PIPELINE SUMMARY")
    print("=" * 70)
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\nTotal Steps: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total Duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    
    print("\n" + "-" * 70)
    print("Step-by-Step Results:")
    print("-" * 70)
    
    for result in results:
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        print(f"[{status}] Step {result['step']:2d}: {result['name']:<35} ({result['duration']:6.2f}s)")
    
    print("\n" + "=" * 70)
    
    if failed > 0:
        print(f"\n⚠ Pipeline completed with {failed} error(s)")
        print("Please check the output above for details.")
        return False
    else:
        print("\n✓ All steps completed successfully!")
        return True


def main():
    """Main pipeline execution"""
    print_header()
    
    results = []
    pipeline_start = time.time()
    
    # Execute each step
    for step_info in PIPELINE_STEPS:
        print_step_header(step_info)
        
        script_path = SRC_DIR / step_info['script']
        
        # Check if script exists
        if not script_path.exists():
            print(f"[ERROR] Script not found: {script_path}")
            results.append({
                'step': step_info['step'],
                'name': step_info['name'],
                'success': False,
                'duration': 0.0
            })
            continue
        
        # Run the script
        success, duration = run_script(script_path, step_info)
        
        results.append({
            'step': step_info['step'],
            'name': step_info['name'],
            'success': success,
            'duration': duration
        })
        
        # Stop pipeline if critical step fails
        if not success and step_info['step'] <= 3:
            print("\n[CRITICAL] Early pipeline step failed. Stopping execution.")
            break
    
    # Print summary
    total_duration = time.time() - pipeline_start
    success = print_summary(results, total_duration)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Pipeline execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n[FATAL ERROR] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
