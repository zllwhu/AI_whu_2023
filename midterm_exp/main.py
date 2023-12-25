from midterm_exp.genetic_algorithm import TSP
from midterm_exp.genetic_algorithm_visualization import Visualizer

if __name__ == '__main__':
    tsp = TSP()
    tsp.run()
    tsp.plot()
    path, distance = tsp.visualize()
    visualizer = Visualizer(path, distance)
    visualizer.plot_cities()
