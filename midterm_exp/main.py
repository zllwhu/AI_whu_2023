from midterm_exp.genetic_algorithm import TSP
from midterm_exp.visualize import Visualization

if __name__ == '__main__':
    tsp = TSP()
    tsp.run()
    tsp.plot()
    path, distance = tsp.visualize()
    visualizer = Visualization(path, distance)
    visualizer.plot_cities()
