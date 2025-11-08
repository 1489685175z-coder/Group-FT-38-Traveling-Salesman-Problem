# Traveling Salesman Problem Solutions

This repository contains multiple algorithms to solve the Traveling Salesman Problem (TSP).

## 🧬 Algorithms

### 1. Genetic Algorithm
**Branch:** `genetic-algorithm-tsp`

A genetic algorithm implementation for solving TSP with features:
- Customizable parameters (population size, mutation rate, etc.)
- Visualization of results
- Convergence tracking

**Files:**
- `src/tsp_ga.py` - Main genetic algorithm implementation
- `examples/tsp_demo.ipynb` - Usage examples
- `requirements.txt` - Dependencies

### 2. Nearest Neighbor Algorithm  
**Branch:** `nearest-neighbor-tsp`

A greedy nearest neighbor approach for TSP with features:
- Simple and fast implementation
- Step-by-step path construction
- Performance comparison

**Files:**
- `src/tsp_nn.py` - Nearest neighbor implementation
- `examples/nn_demo.ipynb` - Usage examples

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
# Clone the repository
git clone https://github.com/Kasimer-0/Group-FT-38-Traveling-Salesman-Problem.git
cd Group-FT-38-Traveling-Salesman-Problem

# Switch to desired algorithm branch
git checkout genetic-algorithm-tsp

# Install dependencies
pip install -r requirements.txt
