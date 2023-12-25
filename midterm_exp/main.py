from midterm_exp.visualizer import Visualizer
from midterm_exp.tsp import TSP

if __name__ == '__main__':
    tsp = TSP()
    tsp.run_ga()
    tsp.plot_ga()
    path, distance = tsp.visualize_ga()
    visualizer = Visualizer(path, distance)
    visualizer.plot_cities()
