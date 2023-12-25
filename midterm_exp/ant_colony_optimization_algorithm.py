class AntColonyOptimizationAlgorithm:
    def __init__(self, ant_num, pheromone_importance, heuristic_importance, pheromone_volatility, pheromone_constant,
                 iteration):
        self.ant_num = ant_num
        self.pheromone_importance = pheromone_importance
        self.heuristic_importance = heuristic_importance
        self.pheromone_volatility = pheromone_volatility
        self.pheromone_constant = pheromone_constant
        self.iteration = iteration
