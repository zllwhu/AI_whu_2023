from midterm_exp.visualizer import Visualizer
from midterm_exp.tsp import TSP

if __name__ == '__main__':
    tsp = TSP()
    """ Genetic Algorithm """
    tsp.run_ga()
    tsp.plot(1)
    path, distance = tsp.visualize_ga()
    visualizer = Visualizer(path, distance)
    visualizer.plot_cities(1)
    """ Ant Colony Optimization Algorithm """
    path, distance = tsp.run_acoa()
    tsp.plot(2)
    visualizer = Visualizer(path, distance)
    visualizer.plot_cities(2)
