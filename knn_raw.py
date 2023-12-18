import numpy as np
import math
from collections import Counter
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import get_time


def euclidean_distance(p1, p2):
    distance = 0.0
    for i in range(len(p1)):
        distance += (p1[i] - p2[i]) ** 2
    return math.sqrt(distance)


class KNN:
    def __init__(self, k, X_train, y_train):
        self.k = k
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        distances = []
        for i in range(len(self.X_train)):
            distance = euclidean_distance(self.X_train[i], X_test)
            distances.append((distance, self.y_train[i]))
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]
        k_nearest_labels = [label for (_, label) in k_nearest]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


data_train = np.loadtxt('./pen+based+recognition+of+handwritten+digits/pendigits.tra', delimiter=',')
data_test = np.loadtxt('./pen+based+recognition+of+handwritten+digits/pendigits.tes', delimiter=',')

X_train = data_train[:, :-1]
y_train = data_train[:, -1]
X_test = data_test[:, :-1]
y_test = data_test[:, -1]

print("训练集样本数：" + str(y_train.size))
print("测试集样本数：" + str(y_test.size))

result_accuracy = []
k = [1, 2, 3, 4, 5, 6, 7]


def draw_fig():
    plt.figure(figsize=(10, 6), dpi=800)
    plt.rcParams['backend'] = 'Agg'
    plt.plot(k, result_accuracy, color='blue', marker='o', linestyle='--', markerfacecolor='blue')
    plt.title("Accuracy for KNN Algorithm using the scikit-learn Library", fontweight='bold')
    plt.xticks([1, 2, 3, 4, 5, 6, 7])
    plt.xlabel("K value", fontweight='bold')
    plt.ylabel("Accuracy", fontweight='bold')
    plt.grid(True, linestyle='dashed')  # 添加网格线
    plt.savefig('figs/knn_raw.png')
    plt.show()


@get_time
def train_and_predict():
    for i in k:
        print("当前 k 值：" + str(i))
        knn = KNN(k=i, X_train=X_train, y_train=y_train)
        y_pred = []
        for j in tqdm(range(len(X_test)), unit='j'):
            pred = knn.predict(X_test[j])
            y_pred.append(pred)
        accuracy = accuracy_score(y_test, y_pred)
        result_accuracy.append(accuracy)
        print("当前精度：" + str(accuracy))


def knn_raw():
    train_and_predict()
    draw_fig()
