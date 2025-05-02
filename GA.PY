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

class GeneticAlgorithm:
    def __init__(self, n_iterations=1000, population_size=100, mutation_rate=0.1, elite_size=5):
        self.n_iterations = n_iterations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
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
            x = np.random.uniform(0, 8)
            y = np.random.uniform(0, 8)
            a = np.random.choice([2, 4, 6])
            population.append((x, y, a))
        return population
    
    def fitness(self, individual):
        """Calculate fitness (negative loss) for an individual"""
        x, y, a = individual
        return -objective_function(x, y, a)  # Negative because we want to maximize fitness
    
    def select_parents(self, population, fitnesses):
        """Select parents using tournament selection"""
        parents = []
        for _ in range(self.population_size - self.elite_size):
            # Tournament selection
            tournament_size = 3
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
            parents.append(population[winner_idx])
        return parents
    
    def crossover(self, parent1, parent2):
        """Perform crossover between two parents"""
        x1, y1, a1 = parent1
        x2, y2, a2 = parent2
        
        # Arithmetic crossover for continuous variables
        alpha = np.random.random()
        child_x = alpha * x1 + (1 - alpha) * x2
        child_y = alpha * y1 + (1 - alpha) * y2
        
        # Random choice for discrete variable
        child_a = np.random.choice([a1, a2])
        
        return (child_x, child_y, child_a)
    
    def mutate(self, individual):
        """Apply mutation to an individual"""
        x, y, a = individual
        
        # Mutate continuous variables with larger step size
        if np.random.random() < self.mutation_rate:
            x += np.random.normal(0, 1.0)
            x = np.clip(x, 0, 6)
        if np.random.random() < self.mutation_rate:
            y += np.random.normal(0, 1.0)
            y = np.clip(y, 0, 4)
            
        # Mutate discrete variable
        if np.random.random() < self.mutation_rate:
            possible_values = [2, 4, 6]
            possible_values.remove(a)
            a = np.random.choice(possible_values)
            
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
            # Calculate fitness for all individuals
            fitnesses = [self.fitness(ind) for ind in population]
            
            # Find best individual
            best_idx = np.argmax(fitnesses)
            best_individual = population[best_idx]
            best_loss = -fitnesses[best_idx]  # Convert back to loss
            
            # Update best solution
            if best_loss < self.best_loss:
                self.best_loss = best_loss
                self.best_params = best_individual
                self.convergence_iteration = iteration
            
            # Select parents
            parents = self.select_parents(population, fitnesses)
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitnesses)[-self.elite_size:]
            new_population.extend([population[i] for i in elite_indices])
            
            # Create offspring
            while len(new_population) < self.population_size:
                # Select parents
                parent1_idx = np.random.randint(len(parents))
                parent2_idx = np.random.randint(len(parents))
                parent1 = parents[parent1_idx]
                parent2 = parents[parent2_idx]
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutation
                child = self.mutate(child)
                
                new_population.append(child)
            
            # Update population
            population = new_population
            
            # Record results
            self.iterations.append(iteration)
            self.losses.append(best_loss)
            self.parameters.append(best_individual)
        
        return self

def run_ga_benchmark(n_runs=30):
    """Run multiple GA optimizations and collect metrics"""
    results = []
    all_iterations = []
    
    for run in range(n_runs):
        # Run GA with different random seeds
        ga = GeneticAlgorithm()
        ga_results = ga.optimize(seed=run)
        
        # Store run results
        results.append({
            'run': run,
            'best_loss': ga_results.best_loss,
            'convergence_iteration': ga_results.convergence_iteration,
            'final_x': ga_results.best_params[0],
            'final_y': ga_results.best_params[1],
            'final_a': ga_results.best_params[2]
        })
        
        # Store iteration data
        run_iterations = pd.DataFrame({
            'Run': run,
            'Iteration': ga_results.iterations,
            'Loss': ga_results.losses,
            'X': [p[0] for p in ga_results.parameters],
            'Y': [p[1] for p in ga_results.parameters],
            'A': [p[2] for p in ga_results.parameters]
        })
        all_iterations.append(run_iterations)
    
    # Combine all results
    results_df = pd.DataFrame(results)
    iterations_df = pd.concat(all_iterations, ignore_index=True)
    
    return results_df, iterations_df

def save_results_to_excel(results_df, iterations_df):
    """Save optimization results to Excel file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ga_results_{timestamp}.xlsx"
    
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
    plt.title('Genetic Algorithm Convergence Plot')
    plt.legend()
    plt.grid(True)
    plt.savefig('ga_convergence_plot.png')
    plt.close()

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def main():
    print("Running Genetic Algorithm optimization...")
    start_time = time.time()
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Run benchmark
    results_df, iterations_df = run_ga_benchmark(n_runs=30)
    
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
