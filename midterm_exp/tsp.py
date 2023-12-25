import math
from matplotlib import pyplot as plt

from midterm_exp.genetic_algorithm import GeneticAlgorithm


class TSP:
    def __init__(self, life_count=100):
        self.cities = []
        self.init_cities()
        self.life_count = life_count
        self.ga = GeneticAlgorithm(across_rate=0.7, mutation_rate=0.02, life_count=self.life_count,
                                   gene_length=len(self.cities), match_fun=self.match_fun_ga())
        self.distance_record = []

    def init_cities(self):
        self.cities.append((116.46, 39.92))
        self.cities.append((117.2, 39.13))
        self.cities.append((121.48, 31.22))
        self.cities.append((106.54, 29.59))
        self.cities.append((91.11, 29.97))
        self.cities.append((87.68, 43.77))
        self.cities.append((106.27, 38.47))
        self.cities.append((111.65, 40.82))
        self.cities.append((108.33, 22.84))
        self.cities.append((126.63, 45.75))
        self.cities.append((125.35, 43.88))
        self.cities.append((123.38, 41.8))
        self.cities.append((114.48, 38.03))
        self.cities.append((112.53, 37.87))
        self.cities.append((101.74, 36.56))
        self.cities.append((117, 36.65))
        self.cities.append((113.6, 34.76))
        self.cities.append((118.78, 32.04))
        self.cities.append((117.27, 31.86))
        self.cities.append((120.19, 30.26))
        self.cities.append((119.3, 26.08))
        self.cities.append((115.89, 28.68))
        self.cities.append((113, 28.21))
        self.cities.append((114.31, 30.52))
        self.cities.append((113.23, 23.16))
        self.cities.append((121.5, 25.05))
        self.cities.append((110.35, 20.02))
        self.cities.append((103.73, 36.03))
        self.cities.append((108.95, 34.27))
        self.cities.append((104.06, 30.67))
        self.cities.append((106.71, 26.57))
        self.cities.append((102.73, 25.04))
        self.cities.append((114.1, 22.2))
        self.cities.append((113.33, 22.13))

    def distance(self, gene):
        distance = 0.0
        for i in range(-1, len(self.cities) - 1):
            index1 = gene[i]
            index2 = gene[i + 1]
            city1 = self.cities[index1]
            city2 = self.cities[index2]
            distance += math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)
        return distance

    def match_fun_ga(self):
        return lambda life: 1.0 / self.distance(life.gene)

    def run_ga(self, max_iter=80, print_process=True):
        while max_iter > 0:
            self.ga.next_iter()
            distance = self.distance(self.ga.best.gene)
            self.distance_record.append(distance)
            if print_process:
                print("Generation: %4d \t\t Distance: %f" % (self.ga.generation - 1, distance))
                print("Optimal path: ", self.ga.best.gene)
            max_iter -= 1

    def plot_ga(self):
        plt.figure(figsize=(10, 6), dpi=800)
        plt.rcParams['backend'] = 'Agg'
        plt.plot(list(range(1, 81)), self.distance_record, color='red')
        plt.title("Minimum Distance Curve", fontweight='bold')
        plt.xlabel('Iteration', fontweight='bold')
        plt.ylabel('Minimum Distance', fontweight='bold')
        plt.grid(True, linestyle='dashed')
        plt.savefig('figs/GA_distance_curve.png')

    def visualize_ga(self):
        path = self.ga.best.gene + [self.ga.best.gene[0]]
        places = ['北京', '天津', '上海', '重庆', '拉萨', '乌鲁木齐', '银川', '呼和浩特', '南宁', '哈尔滨', '长春',
                  '沈阳', '石家庄', '太原', '西宁', '济南', '郑州', '南京', '合肥', '杭州', '福州', '南昌', '长沙',
                  '武汉', '广州', '台北', '海口', '兰州', '西安', '成都', '贵阳', '昆明', '香港', '澳门']
        for i in range(len(path)):
            if i != len(path) - 1:
                print(f'{places[path[i]]}->', end='')
            else:
                print(places[path[i]])
        return path, self.distance(self.ga.best.gene)
