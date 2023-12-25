import math
import random

import numpy as np


def roulette(possibility):
    cumulative_sum_possibilities = np.nancumsum(possibility)
    tmp = np.where(cumulative_sum_possibilities >= np.random.uniform(low=0.0, high=1.0))[0]
    return tmp[0] if len(tmp) > 0 else len(possibility) - 1


class AntColonyOptimizationAlgorithm:
    def __init__(self, ant_num, pheromone_importance, heuristic_importance, pheromone_volatility, pheromone_constant,
                 city_num, distance_matrix):
        self.ant_num = ant_num
        self.pheromone_importance = pheromone_importance
        self.heuristic_importance = heuristic_importance
        self.pheromone_volatility = pheromone_volatility
        self.pheromone_constant = pheromone_constant
        self.city_num = city_num
        self.distance_matrix = distance_matrix
        self.global_minima = 1e8
        self.global_minima_path = np.zeros(self.city_num)
        self.pheromone_matrix = np.ones((self.city_num, self.city_num))
        self.path_matrix = np.zeros((self.ant_num, self.city_num), dtype=np.int64) - 1

    def run(self):
        # 初始化蚂蚁位置
        if self.ant_num <= self.city_num:
            self.path_matrix[:, 0] = np.array(random.sample([i for i in range(self.city_num)], self.ant_num))
        else:
            self.path_matrix[:, 0] = np.array([random.randint(0, self.city_num - 1) for _ in range(self.ant_num)])

        # 蚂蚁开始移动
        local_distance = np.zeros(self.ant_num)
        for k in range(self.ant_num):
            possibility = np.zeros(self.city_num)
            i = -1
            for i in range(self.city_num - 1):
                current_position = self.path_matrix[k, i]
                for next_position in range(self.city_num):
                    if next_position in self.path_matrix[k]:
                        possibility[next_position] = 0
                    else:
                        possibility[next_position] = (math.pow(self.pheromone_matrix[current_position, next_position],
                                                               self.pheromone_importance) /
                                                      math.pow(self.distance_matrix[current_position, next_position],
                                                               self.heuristic_importance))
                possibility = possibility / sum(possibility)
                self.path_matrix[k, i + 1] = roulette(possibility)
                local_distance[k] += self.distance_matrix[self.path_matrix[k, i], self.path_matrix[k, i + 1]]
            local_distance[k] += self.distance_matrix[self.path_matrix[k, i + 1], self.path_matrix[k, 0]]

        # 更新信息素
        increment_pheromone_matrix = np.zeros_like(self.pheromone_matrix)
        for k in range(self.ant_num):
            i = -1
            for i in range(self.city_num - 1):
                increment_pheromone_matrix[
                    self.path_matrix[k, i], self.path_matrix[k, i + 1]] += self.pheromone_constant / local_distance[k]
                increment_pheromone_matrix[
                    self.path_matrix[k, i + 1], self.path_matrix[k, i]] += self.pheromone_constant / local_distance[k]
            increment_pheromone_matrix[
                self.path_matrix[k, i + 1], self.path_matrix[k, 0]] += self.pheromone_constant / local_distance[k]
            increment_pheromone_matrix[
                self.path_matrix[k, 0], self.path_matrix[k, i + 1]] += self.pheromone_constant / local_distance[k]

        # 返回本次迭代结果
        local_best_distance, local_best_path = min(local_distance), self.path_matrix[np.argmin(local_distance)]
        if self.global_minima > local_best_distance:
            self.global_minima = local_best_distance
            self.global_minima_path = local_best_path
        self.path_matrix = np.zeros((self.ant_num, self.city_num), dtype=np.int64) - 1
        return self.global_minima, self.global_minima_path
