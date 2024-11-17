########################################################################################
# DISCLAIMER: AI was used to help debug and generate portions of this code. 
#
# Multi-Objective Evolutionary Algorithm
#
# Generates a best-path through the 2D terrain where it minimizes path length
# and maximizes points
#
# Jack Cooperman, Anuj Suvarna
# November 2024
########################################################################################

import numpy as np
from typing import List, Tuple
import random

class MOEA:
    def __init__(
        self, 
        start: Tuple[float, float],
        end: Tuple[float, float],
        checkpoints: List[Tuple[float, float]],
        destination_radius: float,
        population_size: int = 100,
        num_generations: int = 100,
        mutation_rate: float = 0.1,
        num_weight_vectors: int = 10
    ):
        self.start = np.array(start)
        self.end = np.array(end)
        self.destination_radius = destination_radius
        self.checkpoints = [np.array(cp) for cp in checkpoints]
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.num_weight_vectors = num_weight_vectors
        self.weights = self._generate_weight_vectors()
        
        max_path_length = np.linalg.norm(self.end - self.start) * 2
        max_checkpoint_score = len(self.checkpoints)
        self.reference_point = np.array([max_path_length, 0])  # Initialize for min path, max checkpoint
    
    def _generate_weight_vectors(self) -> np.ndarray:
        weights = []
        for i in range(self.num_weight_vectors):
            w = i / (self.num_weight_vectors - 1)
            weights.append([w, 1 - w])
        return np.array(weights)
    
    def _initialize_population(self) -> List[np.ndarray]:
        population = []
        for _ in range(self.population_size):
            num_waypoints = random.randint(2, max(3, len(self.checkpoints)))
            path = [self.start]
            
            for _ in range(num_waypoints):
                if random.random() < 0.5 and self.checkpoints:
                    checkpoint = random.choice(self.checkpoints)
                    noise = np.random.normal(0, 5, 2)
                    waypoint = checkpoint + noise
                else:
                    x = random.uniform(min(self.start[0], self.end[0]), 
                                     max(self.start[0], self.end[0]))
                    y = random.uniform(min(self.start[1], self.end[1]), 
                                     max(self.start[1], self.end[1]))
                    waypoint = np.array([x, y])
                path.append(waypoint)
                
            path.append(self.end)
            population.append(np.array(path))
        return population
    
    def _evaluate_objectives(self, path: np.ndarray) -> Tuple[float, float]:
        # Path length (minimize)
        distance = 0
        for i in range(len(path) - 1):
            if i == len(path) - 2:
                remaining = path[i+1] - path[i]
                remaining_dist = np.linalg.norm(remaining)
                if remaining_dist > self.destination_radius:
                    distance += remaining_dist - self.destination_radius
            else:
                distance += np.linalg.norm(path[i+1] - path[i])
        
        # Checkpoint coverage (maximize)
        checkpoint_score = 0
        for checkpoint in self.checkpoints:
            min_distance = float('inf')
            for point in path:
                dist = np.linalg.norm(point - checkpoint)
                min_distance = min(min_distance, dist)
            checkpoint_score += 1 / (1 + min_distance)
            
        return distance, checkpoint_score
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        if len(parent1) != len(parent2):
            new_length = max(3, min(len(parent1), len(parent2)))
            child = [self.start]
            
            for i in range(1, new_length-1):
                t = i / (new_length - 1)
                if random.random() < 0.5:
                    idx1 = int(t * (len(parent1)-2)) + 1
                    point = parent1[idx1]
                else:
                    idx2 = int(t * (len(parent2)-2)) + 1
                    point = parent2[idx2]
                child.append(point)
                
            child.append(self.end)
            return np.array(child)
        else:
            child = [self.start]
            for i in range(1, len(parent1)-1):
                if random.random() < 0.5:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])
            child.append(self.end)
            return np.array(child)
    
    def _mutate(self, path: np.ndarray) -> np.ndarray:
        mutated_path = path.copy()
        
        for i in range(1, len(path) - 1):
            if random.random() < self.mutation_rate:
                min_checkpoint_dist = float('inf')
                for checkpoint in self.checkpoints:
                    dist = np.linalg.norm(path[i] - checkpoint)
                    min_checkpoint_dist = min(min_checkpoint_dist, dist)
                
                mutation_scale = min(1.0, min_checkpoint_dist / 10.0)
                offset = np.random.normal(0, mutation_scale, 2)
                mutated_path[i] += offset
                
        return mutated_path
    
    def optimize(self) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
        population = self._initialize_population()
        
        for generation in range(self.num_generations):
            objectives = [self._evaluate_objectives(path) for path in population]
            
            # Update reference points for min path length, max checkpoint score
            min_path = min(obj[0] for obj in objectives)
            max_checkpoint = max(obj[1] for obj in objectives)
            self.reference_point = np.array([min_path, max_checkpoint])
            
            new_population = []
            for _ in range(self.population_size):
                tournament = random.sample(list(enumerate(population)), 3)
                
                best_idx = None
                best_rank = float('inf')
                
                for idx, solution in tournament:
                    rank = 0
                    for other_idx, other_obj in enumerate(objectives):
                        if idx != other_idx:
                            if (other_obj[0] <= objectives[idx][0] and 
                                other_obj[1] >= objectives[idx][1] and
                                (other_obj[0] < objectives[idx][0] or 
                                other_obj[1] > objectives[idx][1])):
                                rank += 1
                
                    if rank < best_rank:
                        best_rank = rank
                        best_idx = idx
                
                parent1 = population[best_idx]
                parent2 = population[random.randint(0, self.population_size-1)]
                
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)
            
            population = new_population
        
        final_objectives = [self._evaluate_objectives(path) for path in population]
        
        # Modified Pareto dominance check with tolerance
        pareto_indices = []
        for i, obj1 in enumerate(final_objectives):
            dominated = False
            for j, obj2 in enumerate(final_objectives):
                if i != j:
                    if (obj2[0] <= obj1[0] and obj2[1] >= obj1[1] and
                                        (obj2[0] < obj1[0] or obj2[1] > obj1[1])):
                                        dominated = True
                                        break
            if not dominated:
                pareto_indices.append(i)

        # # Sort pareto front by path length
        pareto_indices.sort(key=lambda i: final_objectives[i][0])
        
        pareto_paths = [population[i] for i in pareto_indices]
        pareto_objectives = [final_objectives[i] for i in pareto_indices]
        
        return pareto_paths, pareto_objectives