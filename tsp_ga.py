import numpy as np
import matplotlib.pyplot as plt
import random
import time
from typing import List, Tuple

class GeneticAlgorithmTSP:
    def __init__(self, cities: List[Tuple[float, float]], population_size: int = 100, 
                 generations: int = 1000, crossover_rate: float = 0.8, 
                 mutation_rate: float = 0.2, elitism_count: int = 2):
        """
        Problem initiallization
        
        Args:
            cities: List of city coordinates [(x1, y1), (x2, y2), ...]
            population_size: Size of population
            generations: Number of generations to evolve
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            elitism_count: Number of elite individuals to preserve
        """
        self.cities = cities
        self.num_cities = len(cities)
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        
        # Calculate distance matrix between cities
        self.distance_matrix = self._calculate_distance_matrix()

        # Initialize population
        self.population = self._initialize_population()

        # Track best route and distance
        self.best_route = None
        self.best_distance = float('inf')   # Start with infinity
        self.best_distances = []    # For convergence tracking

    def _initialize_population(self):
        """Initialize random population of routes"""
        population = []
        for _ in range(self.population_size):
            # Create base route [0, 1, 2, ..., n-1]
            route = list(range(self.num_cities))
            # Shuffle to create random route
            random.shuffle(route)
            population.append(route)
        return population
        
    def _calculate_distance_matrix(self) -> np.ndarray:
        """Calculate distance matrix between all cities"""
        matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                # Calculate distance
                dist = np.sqrt((self.cities[i][0] - self.cities[j][0])**2 + 
                            (self.cities[i][1] - self.cities[j][1])**2)
                matrix[i][j] = dist
                matrix[j][i] = dist  # Symmetric matrix
        return matrix
        
    def _calculate_route_distance(self, route: List[int]) -> float:
        """Calculate total distance of a route"""
        total_distance = 0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]  # Wrap around to start
            total_distance += self.distance_matrix[from_city][to_city]
        return total_distance
        
    def _fitness(self, route: List[int]) -> float:
        """Calculate fitness as inverse of distance"""
        """shorter distance = higher fitness"""
        distance = self._calculate_route_distance(route)
        return 1.0 / distance
            
    def _selection(self, fitnesses: List[float]) -> List[List[int]]:
        """Selection operation using roulette wheel selection"""
        # Calculate selection probabilities
        total_fitness = sum(fitnesses)
        probabilities = [f / total_fitness for f in fitnesses]
        
        # Roulette wheel selection
        selected = []
        for _ in range(self.population_size - self.elitism_count):
            r = random.random()  # Random numebr 0 - 1
            cumulative_prob = 0
            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if cumulative_prob >= r:
                    selected.append(self.population[i][:])  # Deep copy
                    break
        return selected
        
    def _crossover(self, parent1, parent2):
        """Crossover operation: Order Crossover (OX)"""
        # Skip crossover based on probability
        if random.random() > self.crossover_rate:
            return parent1[:]
        
        size = self.num_cities
        # Select two random cut points
        start, end = sorted(random.sample(range(size), 2))
        
        # Initialize child
        child = [None] * size
        
        # Copy segment from parent1 to child
        child[start:end+1] = parent1[start:end+1]
        
        # Create set for quick lookup
        segment_set = set(child[start:end+1])
        
        # Get remaining cities in parent2 order
        remaining = [gene for gene in parent2 if gene not in segment_set]
        
        # Fill remaining positions
        idx = 0
        pos = (end + 1) % size
        while idx < len(remaining):
            if child[pos] is None:
                child[pos] = remaining[idx]
                idx += 1
            pos = (pos + 1) % size
        
        return child
    
    def _mutation(self, route):
        """Mutation operation: Inversion Mutation"""
        new_route = route[:]  # Copy route
        
        # Skip mutation based on probability
        if random.random() > self.mutation_rate:
            return new_route
        
        # Edge case: need at least 2 cities for mutation
        if self.num_cities < 2:
            return new_route
        
        # Select two random points and reverse segment
        start, end = sorted(random.sample(range(self.num_cities), 2))
        new_route[start:end+1] = new_route[start:end+1][::-1]
        
        return new_route
        
    def _elitism(self, fitnesses: List[float]) -> List[List[int]]:
        """Elitism: preserve top individuals"""
        # Get indices of top fitness individuals
        elite_indices = np.argsort(fitnesses)[-self.elitism_count:]
        elites = [self.population[i][:] for i in elite_indices]
        return elites
        
    def evolve(self) -> Tuple[List[int], float]:
        """Run genetic algorithm evolution process"""
        print("Starting genetic algorithm for TSP...")
        print(f"Cities: {self.num_cities}, Population: {self.population_size}, Generations: {self.generations}")
        
        start_time = time.time()
        
        for generation in range(self.generations):
            # 1. Calculate fitness
            fitnesses = [self._fitness(route) for route in self.population]
            
            # 2. Elitism
            elites = self._elitism(fitnesses)
            
            # 3. Selection
            selected = self._selection(fitnesses)
            
            # 4. Crossover and mutation for new population
            new_population = elites
            
            # Apply crossover and mutation to selected individuals
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    # Crossover (call twice with swapped parents)
                    child1 = self._crossover(selected[i], selected[i + 1])
                    child2 = self._crossover(selected[i + 1], selected[i])
                    
                    # Mutation
                    child1 = self._mutation(child1)
                    child2 = self._mutation(child2)
                    
                    new_population.append(child1)
                    new_population.append(child2)
            
            # Handle odd number of selected individuals
            if len(selected) % 2 != 0:
                child = self._mutation(selected[-1])
                new_population.append(child)
            
            self.population = new_population
            
            # 5. Find best in current population
            current_distances = [self._calculate_route_distance(route) for route in self.population]
            current_best_index = np.argmin(current_distances)
            current_best_route = self.population[current_best_index]
            current_best_distance = current_distances[current_best_index]
            
            # 6. Update global best
            if current_best_distance < self.best_distance:
                self.best_distance = current_best_distance
                self.best_route = current_best_route[:]  # Deep copy
            
            # 7. Track convergence
            self.best_distances.append(self.best_distance)
            
            # Print progress every 100 generations
            if generation % 100 == 0:
                print(f"Generation {generation}: Best distance = {self.best_distance:.2f}")
        
        end_time = time.time()
        print(f"\nAlgorithm completed! Total time: {end_time - start_time:.2f} seconds")
        print(f"Best route distance: {self.best_distance:.2f}")
        
        return self.best_route, self.best_distance

    def plot_results(self):
        """
        Plot genetic algorithm results:
        1. Best route visualization
        2. Convergence curve
        """
        plt.figure(figsize=(12, 5))

        # ===== Subplot 1: Route visualization =====
        plt.subplot(1, 2, 1)

        # Extract coordinates for best route
        xs = [self.cities[i][0] for i in self.best_route] + [self.cities[self.best_route[0]][0]]
        ys = [self.cities[i][1] for i in self.best_route] + [self.cities[self.best_route[0]][1]]

        plt.plot(xs, ys, 'o-', markersize=8, linewidth=2)
        plt.scatter(xs[0], ys[0], color='red', s=100, label='Start', zorder=5)
        
        # Add city labels
        for i, (x, y) in enumerate(self.cities):
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
            
        plt.title(f'Best Route (Distance: {self.best_distance:.2f})')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # ===== Subplot 2: Convergence curve =====
        plt.subplot(1, 2, 2)
        plt.plot(self.best_distances, linewidth=2)
        plt.title('Convergence Curve')
        plt.xlabel('Generation')
        plt.ylabel('Best Distance')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def generate_random_cities(num_cities: int, x_range: Tuple[float, float] = (0, 100), 
                          y_range: Tuple[float, float] = (0, 100)) -> List[Tuple[float, float]]:
    """Generate random city coordinates"""
    cities = []
    for _ in range(num_cities):
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        cities.append((x, y))
    return cities