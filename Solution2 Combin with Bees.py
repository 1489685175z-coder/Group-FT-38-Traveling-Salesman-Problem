import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time
import math

class TSPSolver:
    def __init__(self):
        self.distance_matrix = None
        self.cities = None
        self.best_tour = None
        self.best_distance = float('inf')
    
    def load_tsp_data(self, file_path: str) -> Dict:
        """Load data in TSPLIB format"""
        cities = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        node_coord_section = False
        for line in lines:
            if line.startswith('DIMENSION'):
                dimension = int(line.split(':')[1].strip())
            elif line.startswith('NODE_COORD_SECTION'):
                node_coord_section = True
                continue
            elif line.startswith('EOF'):
                break
            elif node_coord_section:
                parts = line.strip().split()
                if len(parts) == 3:
                    city_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    cities.append((x, y))
        
        self.cities = cities
        self._create_distance_matrix()
        return {'dimension': dimension, 'cities': cities}
    
    def _create_distance_matrix(self):
        """Create distance matrix"""
        n = len(self.cities)
        self.distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = self.cities[i]
                x2, y2 = self.cities[j]
                distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                self.distance_matrix[i][j] = distance
                self.distance_matrix[j][i] = distance
    
    def nearest_neighbor(self, start_city: int = 0) -> Tuple[List[int], float]:
        "Nearest Neighbor Algorithm Implementation"
        n = len(self.cities)
        unvisited = set(range(n))
        tour = [start_city]
        unvisited.remove(start_city)
        total_distance = 0.0
        
        current_city = start_city
        
        while unvisited:
            # Find the nearest unvisited city
            nearest_city = None
            min_distance = float('inf')
            
            for city in unvisited:
                if self.distance_matrix[current_city][city] < min_distance:
                    min_distance = self.distance_matrix[current_city][city]
                    nearest_city = city
            
            # Move to the nearest city
            tour.append(nearest_city)
            unvisited.remove(nearest_city)
            total_distance += min_distance
            current_city = nearest_city
        
        # Return to starting point
        total_distance += self.distance_matrix[tour[-1]][tour[0]]
        
        if total_distance < self.best_distance:
            self.best_tour = tour
            self.best_distance = total_distance
            
        return tour, total_distance
    
    def multi_start_nearest_neighbor(self, num_starts: int = 10) -> Tuple[List[int], float]:
        "Multi-starting-point nearest neighbor algorithm improves solution quality."
        best_tour = None
        best_distance = float('inf')
        
        for start in range(min(num_starts, len(self.cities))):
            tour, distance = self.nearest_neighbor(start)
            if distance < best_distance:
                best_tour = tour
                best_distance = distance
        
        self.best_tour = best_tour
        self.best_distance = best_distance
        return best_tour, best_distance
    
    def calculate_tour_distance(self, tour: List[int]) -> float:
        """Calculate the total path distance."""
        distance = 0.0
        n = len(tour)
        for i in range(n):
            j = (i + 1) % n
            distance += self.distance_matrix[tour[i]][tour[j]]
        return distance
    
    def plot_tour(self, tour: List[int], title: str = "TSP Tour"):
        """Visual Path"""
        x = [self.cities[i][0] for i in tour]
        y = [self.cities[i][1] for i in tour]
        x.append(x[0])  # Back to the starting point
        y.append(y[0])
        
        plt.figure(figsize=(12, 8))
        plt.plot(x, y, 'o-', markersize=8, linewidth=2)
        plt.title(f"{title}\nTotal Distance: {self.calculate_tour_distance(tour):.2f}")
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        
        # Mark the starting point
        plt.plot(x[0], y[0], 'ro', markersize=10, label='Start')
        plt.legend()
        plt.show()

class CombinatorialBeesAlgorithm:
    """Combining nearest neighbor methods with the bee algorithm"""
    
    def __init__(self, distance_matrix, num_scout_bees=20, num_elite_sites=5,
                 num_selected_sites=20, nep=200, nsp=100, max_iterations=1000):
        self.distance_matrix = distance_matrix
        self.n = len(distance_matrix)
        self.num_scout_bees = num_scout_bees
        self.num_elite_sites = num_elite_sites
        self.num_selected_sites = num_selected_sites
        self.nep = nep  # Bee count around the elite site
        self.nsp = nsp  # Select the number of bees around the site
        self.max_iterations = max_iterations
        self.neighborhood_size = 10  # Neighborhood size
        
        self.best_solution = None
        self.best_distance = float('inf')
        self.population = []
    
    def initialize_population_with_nnm(self):
        """Initialize the population using the nearest neighbor method."""
        print("Initialize the population using NNM..")
        self.population = []
        
        for i in range(self.num_scout_bees):
            # Create NNM solutions using different starting points.
            solver = TSPSolver()
            solver.distance_matrix = self.distance_matrix
            solver.cities = [(0,0)] * self.n  # placeholder
            
            tour, distance = solver.nearest_neighbor(i % self.n)
            self.population.append((tour, distance))
            
            if distance < self.best_distance:
                self.best_solution = tour.copy()
                self.best_distance = distance
    
    def local_search_insert(self, tour, ngh_size=10):
        """Insertion operation local search"""
        if len(tour) <= 3:
            return tour, self.calculate_distance(tour)
        
        improved = True
        best_tour = tour.copy()
        best_distance = self.calculate_distance(tour)
        
        while improved:
            improved = False
            for i in range(len(tour)):
                # Select the insertion position (limited to within the neighborhood).
                insert_positions = list(range(max(0, i-ngh_size), min(len(tour), i+ngh_size+1)))
                if i in insert_positions:
                    insert_positions.remove(i)
                
                for j in insert_positions:
                    if j < 0 or j >= len(tour) or j == i:
                        continue
                    
                    new_tour = best_tour.copy()
                    city = new_tour.pop(i)
                    new_tour.insert(j, city)
                    new_distance = self.calculate_distance(new_tour)
                    
                    if new_distance < best_distance:
                        best_tour = new_tour
                        best_distance = new_distance
                        improved = True
                        break
                if improved:
                    break
        
        return best_tour, best_distance
    
    def local_search_reversion(self, tour, ngh_size=10):
        """Reverse operation local search"""
        best_tour = tour.copy()
        best_distance = self.calculate_distance(tour)
        
        for i in range(len(tour)):
            for j in range(i+2, min(len(tour), i+ngh_size+1)):
                new_tour = tour.copy()
                # Reverse the city order between i and j
                new_tour[i:j] = reversed(new_tour[i:j])
                new_distance = self.calculate_distance(new_tour)
                
                if new_distance < best_distance:
                    best_tour = new_tour
                    best_distance = new_distance
        
        return best_tour, best_distance
    
    def calculate_distance(self, tour):
        """Calculate path distance"""
        distance = 0.0
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            distance += self.distance_matrix[tour[i]][tour[j]]
        return distance
    
    def run(self):
        """Running the bee algorithm"""
        print("Begin combining the bee algorithm....")
        self.initialize_population_with_nnm()
        
        for iteration in range(self.max_iterations):
            # Population assessment
            self.population.sort(key=lambda x: x[1])
            
            # Choose elite sites and regular sites
            elite_sites = self.population[:self.num_elite_sites]
            selected_sites = self.population[:self.num_selected_sites]
            
            new_population = []
            
            # Intensive search around the elite site
            for site in elite_sites:
                tour, distance = site
                for _ in range(self.nep // 10):  # Simplify search times
                    # Randomly select local search operation
                    if np.random.random() < 0.5:
                        new_tour, new_distance = self.local_search_insert(tour, self.neighborhood_size)
                    else:
                        new_tour, new_distance = self.local_search_reversion(tour, self.neighborhood_size)
                    
                    new_population.append((new_tour, new_distance))
                    
                    if new_distance < self.best_distance:
                        self.best_solution = new_tour.copy()
                        self.best_distance = new_distance
            
            # Select the area around the site for a general search.
            for site in selected_sites:
                tour, distance = site
                for _ in range(self.nsp // 10):
                    if np.random.random() < 0.5:
                        new_tour, new_distance = self.local_search_insert(tour, self.neighborhood_size)
                    else:
                        new_tour, new_distance = self.local_search_reversion(tour, self.neighborhood_size)
                    
                    new_population.append((new_tour, new_distance))
            
            # Update population
            self.population = sorted(new_population, key=lambda x: x[1])[:self.num_scout_bees]
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, optimal distance: {self.best_distance:.2f}")
        
        print(f"The algorithm is complete, and the final optimal distance is obtained: {self.best_distance:.2f}")
        return self.best_solution, self.best_distance


# Test function
def complete_example():
    """Complete example: NNM + Bee Algorithm"""
    # Create test data
    n_cities = 30
    np.random.seed(42)
    cities = [(np.random.uniform(0, 100), np.random.uniform(0, 100)) for _ in range(n_cities)]
    
    # Create distance matrix
    solver = TSPSolver()
    solver.cities = cities
    solver._create_distance_matrix()
    
    print("=== Phase 1: Nearest Neighbor Algorithm ===")
    start_time = time.time()
    nnm_tour, nnm_distance = solver.multi_start_nearest_neighbor(5)
    nnm_time = time.time() - start_time
    
    print(f"NNM optimal distance: {nnm_distance:.2f}")
    print(f"NNM computation time: {nnm_time:.4f} seconds")
    
    print("\n=== Phase 2: Combinatorial Bee Algorithm ===")
    cba = CombinatorialBeesAlgorithm(
        distance_matrix=solver.distance_matrix,
        num_scout_bees=20,
        num_elite_sites=5,
        num_selected_sites=20,
        nep=200,
        nsp=100,
        max_iterations=200
    )
    
    start_time = time.time()
    cba_tour, cba_distance = cba.run()
    cba_time = time.time() - start_time
    
    print(f"CBA optimal distance: {cba_distance:.2f}")
    print(f"CBA calculation time: {cba_time:.4f} seconds")
    print(f"Improvement: {nnm_distance - cba_distance:.2f} ({((nnm_distance - cba_distance)/nnm_distance)*100:.1f}%)")
    
    # Visual comparison
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    x = [cities[i][0] for i in nnm_tour]
    y = [cities[i][1] for i in nnm_tour]
    x.append(x[0]); y.append(y[0])
    plt.plot(x, y, 'o-', markersize=6)
    plt.title(f'NNM Solution\nDistance: {nnm_distance:.2f}')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    x = [cities[i][0] for i in cba_tour]
    y = [cities[i][1] for i in cba_tour]
    x.append(x[0]); y.append(y[0])
    plt.plot(x, y, 'o-', markersize=6, color='green')
    plt.title(f'CBA Solution\nDistance: {cba_distance:.2f}')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    complete_example()
