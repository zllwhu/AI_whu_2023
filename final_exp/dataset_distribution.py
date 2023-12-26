from matplotlib import pyplot as plt


class DatasetDistribution:
    def __init__(self):
        self.train_data = [780, 779, 780, 719, 780, 720, 720, 778, 719, 719]
        self.train_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.test_data = [363, 364, 364, 336, 364, 335, 336, 364, 336, 336]
        self.test_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def plot_distribution(self, type):
        plt.figure(figsize=(10, 6), dpi=800)
        plt.rcParams['backend'] = 'Agg'
        plt.xlabel('Number of Instances', fontweight='bold')
        plt.ylabel('Labels', fontweight='bold')
        plt.grid(True, linestyle='dashed')

        if type == 1:
            plt.title("Data Distribution (Train)", fontweight='bold')
            plt.bar(self.train_label, self.train_data, color='blue')
            plt.savefig('figs/train_data_distribution.png')
        elif type == 2:
            plt.title("Data Distribution (Test)", fontweight='bold')
            plt.bar(self.test_label, self.test_data, color='blue')
            plt.savefig('figs/test_data_distribution.png')


if __name__ == '__main__':
    dataset_distribution = DatasetDistribution()
    dataset_distribution.plot_distribution(1)
    dataset_distribution.plot_distribution(2)
