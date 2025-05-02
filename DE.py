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

class DifferentialEvolution:
    def __init__(self, n_iterations=1000, population_size=100, F=0.5, CR=0.7):
        self.n_iterations = n_iterations
        self.population_size = population_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.iterations = []
        self.losses = []
        self.parameters = []
        self.best_loss = float('inf')
        self.best_params = None
        self.convergence_iteration = None
        
    def initialize_population(self):
        """Initialize population with random values"""
        population = []
        for _ in range(self.population_size):
            x = np.random.uniform(0, 8)  # Consistent range with other algorithms
            y = np.random.uniform(0, 8)  # Consistent range with other algorithms
            a = np.random.choice([2, 4, 6])
            population.append((x, y, a))
        return population
    
    def select_vectors(self, population, current_idx):
        """Select three random vectors different from the current vector"""
        indices = list(range(len(population)))
        indices.remove(current_idx)
        selected = np.random.choice(indices, 3, replace=False)
        return [population[i] for i in selected]
    
    def mutate(self, target, vectors):
        """Perform differential mutation"""
        v1, v2, v3 = vectors
        x = v1[0] + self.F * (v2[0] - v3[0])
        y = v1[1] + self.F * (v2[1] - v3[1])
        a = v1[2]  # Keep discrete parameter unchanged during mutation
        
        # Ensure bounds
        x = np.clip(x, 0, 6)
        y = np.clip(y, 0, 4)
        
        return (x, y, a)
    
    def crossover(self, target, mutant):
        """Perform binomial crossover"""
        if np.random.random() < self.CR:
            x = mutant[0]
        else:
            x = target[0]
            
        if np.random.random() < self.CR:
            y = mutant[1]
        else:
            y = target[1]
            
        # For discrete parameter, use random choice between target and mutant
        if np.random.random() < self.CR:
            a = mutant[2]
        else:
            a = target[2]
            
        return (x, y, a)
    
    def optimize(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        # Initialize population
        population = self.initialize_population()
        
        # Reset tracking variables
        self.iterations = []
        self.losses = []
        self.parameters = []
        self.best_loss = float('inf')
        self.best_params = None
        self.convergence_iteration = None
        
        for iteration in range(self.n_iterations):
            new_population = []
            
            # Process each individual in the population
            for i in range(self.population_size):
                target = population[i]
                
                # Select vectors for mutation
                vectors = self.select_vectors(population, i)
                
                # Create mutant vector
                mutant = self.mutate(target, vectors)
                
                # Perform crossover
                trial = self.crossover(target, mutant)
                
                # Selection
                target_loss = objective_function(*target)
                trial_loss = objective_function(*trial)
                
                if trial_loss < target_loss:
                    new_population.append(trial)
                    current_loss = trial_loss
                else:
                    new_population.append(target)
                    current_loss = target_loss
                
                # Update best solution
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.best_params = new_population[-1]
                    self.convergence_iteration = iteration
            
            # Update population
            population = new_population
            
            # Record results
            self.iterations.append(iteration)
            self.losses.append(self.best_loss)
            self.parameters.append(self.best_params)
        
        return self

def run_de_benchmark(n_runs=30):
    """Run multiple DE optimizations and collect metrics"""
    results = []
    all_iterations = []
    
    for run in range(n_runs):
        # Run DE with different random seeds
        de = DifferentialEvolution()
        de_results = de.optimize(seed=run)
        
        # Store run results
        results.append({
            'run': run,
            'best_loss': de_results.best_loss,
            'convergence_iteration': de_results.convergence_iteration,
            'final_x': de_results.best_params[0],
            'final_y': de_results.best_params[1],
            'final_a': de_results.best_params[2]
        })
        
        # Store iteration data
        run_iterations = pd.DataFrame({
            'Run': run,
            'Iteration': de_results.iterations,
            'Loss': de_results.losses,
            'X': [p[0] for p in de_results.parameters],
            'Y': [p[1] for p in de_results.parameters],
            'A': [p[2] for p in de_results.parameters]
        })
        all_iterations.append(run_iterations)
    
    # Combine all results
    results_df = pd.DataFrame(results)
    iterations_df = pd.concat(all_iterations, ignore_index=True)
    
    return results_df, iterations_df

def save_results_to_excel(results_df, iterations_df):
    """Save optimization results to Excel file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"de_results_{timestamp}.xlsx"
    
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
    plt.title('Differential Evolution Convergence Plot')
    plt.legend()
    plt.grid(True)
    plt.savefig('de_convergence_plot.png')
    plt.close()

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def main():
    print("Running Differential Evolution optimization...")
    start_time = time.time()
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Run benchmark
    results_df, iterations_df = run_de_benchmark(n_runs=30)
    
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
