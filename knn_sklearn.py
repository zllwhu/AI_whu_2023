import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from utils import get_time

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
    plt.plot(k, result_accuracy, color='blue', marker='o', linestyle='--')
    plt.title("Accuracy for KNN Algorithm using the scikit-learn Library", fontweight='bold')
    plt.xticks([1, 2, 3, 4, 5, 6, 7])
    plt.xlabel("K value", fontweight='bold')
    plt.ylabel("Accuracy", fontweight='bold')
    plt.grid(True, linestyle='dashed')  # 添加网格线
    # plt.savefig('figs/knn_sklearn.png')
    plt.show()


@get_time
def train_and_predict():
    for i in k:
        print("当前 k 值：" + str(i))
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        result_accuracy.append(accuracy)
        print("当前精度：" + str(accuracy))


def knn_sklearn():
    train_and_predict()
    draw_fig()
