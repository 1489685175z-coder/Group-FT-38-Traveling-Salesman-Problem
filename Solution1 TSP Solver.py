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

# Test code
def test_nnm():
    """Testing the nearest neighbor algorithm"""
    solver = TSPSolver()
    
    # Creating sample data (a small-scale version of the Berlin Problem 52)
    # In actual use, data can be loaded from TSPLIB.
    example_cities = [
        (565.0, 575.0), (25.0, 185.0), (345.0, 750.0), (945.0, 685.0),
        (845.0, 655.0), (880.0, 660.0), (25.0, 230.0), (525.0, 1000.0),
        (580.0, 1175.0), (650.0, 1130.0), (1605.0, 620.0), (1220.0, 580.0),
        (1465.0, 200.0), (1530.0, 5.0), (845.0, 680.0), (725.0, 370.0),
        (145.0, 665.0), (415.0, 635.0), (510.0, 875.0), (560.0, 365.0)
    ]
    
    solver.cities = example_cities
    solver._create_distance_matrix()
    
    print("=== Nearest Neighbor Algorithm Test ===")
    start_time = time.time()
    tour, distance = solver.multi_start_nearest_neighbor(5)
    end_time = time.time()
    
    print(f"Optimal path: {tour}")
    print(f"Total distance: {distance:.2f}")
    print(f"Calculation time: {end_time - start_time:.4f} 秒")
    
    solver.plot_tour(tour, "Nearest Neighbor Solution")

if __name__ == "__main__":
    test_nnm()
