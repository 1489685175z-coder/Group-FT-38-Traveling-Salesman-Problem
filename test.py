import random
import numpy as np
from tsp_ga import GeneticAlgorithmTSP, generate_random_cities


def main():
    # Set random seed if reproducibility needed
    """
    random.seed(42)
    np.random.seed(42)
    """
    
    # Generate random cities
    cities = generate_random_cities(15)
    
    # Create genetic algorithm instance
    ga = GeneticAlgorithmTSP(
        cities=cities,
        population_size=100,
        generations=500,
        crossover_rate=0.85,
        mutation_rate=0.15,
        elitism_count=2
    )
    
    # Run algorithm
    best_route, best_distance = ga.evolve()
    
    # Display output
    print(f"\nBest route: {best_route}")
    print(f"Best distance: {best_distance:.2f}")
    
    # Plot results
    ga.plot_results()

if __name__ == "__main__":
    main()