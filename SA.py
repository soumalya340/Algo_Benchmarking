import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import openpyxl
from datetime import datetime
import time
import psutil
import os

def objective_function(x, y, a):
    """Calculate the objective function L = (x - 3)² + (y - 2)² + |a·x + y - 10|·3 + |a - 4|·x + 5"""
    return (x - 3)**2 + (y - 2)**2 + abs(a*x + y - 10)*3 + abs(a - 4)*x + 5

class SimulatedAnnealing:
    def __init__(self, n_iterations=1000, initial_temp=100, cooling_rate=0.95):
        self.n_iterations = n_iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.iterations = []
        self.losses = []
        self.parameters = []
        self.best_loss = float('inf')
        self.best_params = None
        self.convergence_iteration = None
        
    def optimize(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        # Initialize parameters with better ranges
        # x and y closer to optimal values (near 2-3)
        x = np.random.uniform(0, 8)
        y = np.random.uniform(0, 8)
        # a must be one of the discrete values: 2, 4, or 6
        a = np.random.choice([2, 4, 6])
        
        current_loss = objective_function(x, y, a)
        temperature = self.initial_temp
        
        # Reset tracking variables
        self.iterations = []
        self.losses = []
        self.parameters = []
        self.best_loss = float('inf')
        self.best_params = None
        self.convergence_iteration = None
        
        for iteration in range(self.n_iterations):
            # Generate new solution
            # Adjust step size based on temperature
            step_size = max(0.5, temperature/20)
            
            # Update x and y with scaled random adjustments
            new_x = x + np.random.normal(0, step_size)
            new_y = y + np.random.normal(0, step_size)
            
            # For 'a', randomly select a different value from [2, 4, 6]
            # Sometimes keep the same value (30% chance)
            if np.random.random() < 0.3:
                new_a = a
            else:
                # Get other possible values
                possible_values = [2, 4, 6]
                possible_values.remove(a)
                new_a = np.random.choice(possible_values)
            
            new_loss = objective_function(new_x, new_y, new_a)
            
            # Accept or reject new solution
            if new_loss < current_loss or np.random.random() < np.exp((current_loss - new_loss) / temperature):
                x, y, a = new_x, new_y, new_a
                current_loss = new_loss
                
                # Update convergence iteration if this is the best solution so far
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.best_params = [x, y, a]
                    self.convergence_iteration = iteration
            
            # Update temperature
            temperature *= self.cooling_rate
            
            # Record results
            self.iterations.append(iteration)
            self.losses.append(current_loss)
            self.parameters.append([x, y, a])
        
        return self

def run_sa_benchmark(n_runs=30):
    """Run multiple SA optimizations and collect metrics"""
    results = []
    all_iterations = []
    
    for run in range(n_runs):
        # Run SA with different random seeds
        sa = SimulatedAnnealing()
        sa_results = sa.optimize(seed=run)
        
        # Store run results
        results.append({
            'run': run,
            'best_loss': sa_results.best_loss,
            'convergence_iteration': sa_results.convergence_iteration,
            'final_x': sa_results.best_params[0],
            'final_y': sa_results.best_params[1],
            'final_a': sa_results.best_params[2]
        })
        
        # Store iteration data
        run_iterations = pd.DataFrame({
            'Run': run,
            'Iteration': sa_results.iterations,
            'Loss': sa_results.losses,
            'X': [p[0] for p in sa_results.parameters],
            'Y': [p[1] for p in sa_results.parameters],
            'A': [p[2] for p in sa_results.parameters]
        })
        all_iterations.append(run_iterations)
    
    # Combine all results
    results_df = pd.DataFrame(results)
    iterations_df = pd.concat(all_iterations, ignore_index=True)
    
    return results_df, iterations_df

def save_results_to_excel(results_df, iterations_df):
    """Save optimization results to Excel file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sa_results_{timestamp}.xlsx"
    
    with pd.ExcelWriter(filename) as writer:
        # Save iteration data
        iterations_df.to_excel(writer, sheet_name='Iterations', index=False)
        
        # Calculate and save performance metrics
        metrics = {
            'Metric': [
                'Mean Loss',
                'Std Loss',
                'Mean Convergence Iteration',
                'Std Convergence Iteration',
                'Mean Final X',
                'Std Final X',
                'Mean Final Y',
                'Std Final Y',
                'Mean Final A',
                'Std Final A',
                'A=2 Percentage',
                'A=4 Percentage',
                'A=6 Percentage'
            ],
            'Value': [
                results_df['best_loss'].mean(),
                results_df['best_loss'].std(),
                results_df['convergence_iteration'].mean(),
                results_df['convergence_iteration'].std(),
                results_df['final_x'].mean(),
                results_df['final_x'].std(),
                results_df['final_y'].mean(),
                results_df['final_y'].std(),
                results_df['final_a'].mean(),
                results_df['final_a'].std(),
                100 * (results_df['final_a'] == 2).mean(),
                100 * (results_df['final_a'] == 4).mean(),
                100 * (results_df['final_a'] == 6).mean()
            ]
        }
        pd.DataFrame(metrics).to_excel(writer, sheet_name='Performance Metrics', index=False)

def plot_convergence(iterations_df):
    """Create convergence plot"""
    plt.figure(figsize=(12, 6))
    
    # Plot average loss across runs
    avg_loss = iterations_df.groupby('Iteration')['Loss'].mean()
    plt.plot(avg_loss.index, avg_loss.values, label='Average Loss')
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Simulated Annealing Convergence Plot')
    plt.legend()
    plt.grid(True)
    plt.savefig('sa_convergence_plot.png')
    plt.close()

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def main():
    print("Running Simulated Annealing optimization...")
    start_time = time.time()
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Run benchmark
    results_df, iterations_df = run_sa_benchmark(n_runs=30)
    
    # Calculate percentage of runs finding optimal a=4
    a_equals_4_percent = 100 * (results_df['final_a'] == 4).mean()
    
    # Save results
    save_results_to_excel(results_df, iterations_df)
    
    # Create convergence plot
    plot_convergence(iterations_df)
    
    # Print summary statistics
    print("\nPerformance Metrics:")
    print(f"Mean Loss: {results_df['best_loss'].mean():.4f}")
    print(f"Standard Deviation: {results_df['best_loss'].std():.4f}")
    print(f"Mean Convergence Iteration: {results_df['convergence_iteration'].mean():.1f}")
    print(f"Std Convergence Iteration: {results_df['convergence_iteration'].std():.1f}")
    
    print("\nParameter Stability:")
    print(f"X: mean={results_df['final_x'].mean():.4f}, std={results_df['final_x'].std():.4f}")
    print(f"Y: mean={results_df['final_y'].mean():.4f}, std={results_df['final_y'].std():.4f}")
    print(f"A: mean={results_df['final_a'].mean():.4f}, std={results_df['final_a'].std():.4f}")
    
    print("\nDiscrete Parameter Distribution:")
    print(f"A=2: {100 * (results_df['final_a'] == 2).mean():.1f}%")
    print(f"A=4: {a_equals_4_percent:.1f}%")
    print(f"A=6: {100 * (results_df['final_a'] == 6).mean():.1f}%")
    
    end_time = time.time()
    final_memory = get_memory_usage()
    print(f"\nMemory Usage:")
    print(f"Initial: {initial_memory:.2f} MB")
    print(f"Final: {final_memory:.2f} MB")
    print(f"Peak: {final_memory - initial_memory:.2f} MB")
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()