import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


class Visualizer:
    def __init__(self, best_path, best_distance):
        self.path = best_path
        self.distance = best_distance
        self.cities_location = [(116.46, 39.92), (117.2, 39.13), (121.48, 31.22), (106.54, 29.59), (91.11, 29.97),
                                (87.68, 43.77), (106.27, 38.47), (111.65, 40.82), (108.33, 22.84), (126.63, 45.75),
                                (125.35, 43.88), (123.38, 41.8), (114.48, 38.03), (112.53, 37.87), (101.74, 36.56),
                                (117.0, 36.65), (113.6, 34.76), (118.78, 32.04), (117.27, 31.86), (120.19, 30.26),
                                (119.3, 26.08), (115.89, 28.68), (113.0, 28.21), (114.31, 30.52), (113.23, 23.16),
                                (121.5, 25.05), (110.35, 20.02), (103.73, 36.03), (108.95, 34.27), (104.06, 30.67),
                                (106.71, 26.57), (102.73, 25.04), (114.1, 22.2), (113.33, 22.13)]
        self.cities_name = ['北京', '天津', '上海', '重庆', '拉萨', '乌鲁木齐', '银川', '呼和浩特', '南宁', '哈尔滨',
                            '长春', '沈阳', '石家庄', '太原', '西宁', '济南', '郑州', '南京', '合肥', '杭州', '福州',
                            '南昌', '长沙', '武汉', '广州', '台北', '海口', '兰州', '西安', '成都', '贵阳', '昆明',
                            '香港', '澳门']

    def plot_cities(self, alg):
        plt.rcParams['backend'] = 'Agg'
        font = FontProperties(fname="/System/Library/Fonts/PingFang.ttc")
        longitude = [location[0] for location in self.cities_location]
        latitude = [location[1] for location in self.cities_location]
        fig, ax = plt.subplots(figsize=(10, 6), dpi=800)
        ax.scatter(longitude, latitude, color='blue', zorder=2)
        for i, name in enumerate(self.cities_name):
            ax.annotate(name, (longitude[i], latitude[i]), fontproperties=font, zorder=3)
        for i in range(len(self.path) - 1):
            start = self.path[i]
            end = self.path[i + 1]
            start_location = self.cities_location[start]
            end_location = self.cities_location[end]
            ax.plot([start_location[0], end_location[0]], [start_location[1], end_location[1]], color='red', zorder=1)
        ax.set_xlim(85, 130)
        ax.set_ylim(17, 50)
        plt.text(113, 18, 'Distance = ' + str(self.distance), color='red', fontweight='bold', zorder=4)
        plt.xlabel('Longitude', fontweight='bold')
        plt.ylabel('Latitude', fontweight='bold')
        plt.grid(True, linestyle='dashed')
        if alg == 1:
            plt.title("TSP (Genetic Algorithm)", fontweight='bold')
            plt.savefig('figs/GA_visualize.png')
        elif alg == 2:
            plt.title("TSP (Ant Colony Optimization Algorithm)", fontweight='bold')
            plt.savefig('figs/ACOA_visualize.png')


if __name__ == '__main__':
    path = [25, 20, 21, 16, 13, 7, 12, 15, 1, 0, 9, 10, 11, 2, 19, 17, 18, 23, 22, 28, 6, 27, 14, 5, 4, 31, 29, 3,
            30, 8, 26, 33, 32, 24, 25]
    visualizer = Visualizer(path, 156)
    visualizer.plot_cities()
