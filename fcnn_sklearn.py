import numpy as np
from sklearn.neural_network import MLPClassifier
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

result_accuracy = [[] for _ in range(4)]
hidden_layer_neural_unit = [500, 1000, 1500, 2000]
learning_rate = [0.1, 0.01, 0.001, 0.0001]


def draw_fig():
    plt.figure(figsize=(10, 6), dpi=800)
    plt.rcParams['backend'] = 'Agg'

    plt.plot(hidden_layer_neural_unit, result_accuracy[0], color='blue', marker='o', linestyle='--',
             label='learning rate = 0.1')
    plt.plot(hidden_layer_neural_unit, result_accuracy[1], color='green', marker='s', linestyle='--',
             label='learning rate = 0.01')
    plt.plot(hidden_layer_neural_unit, result_accuracy[2], color='red', marker='^', linestyle='--',
             label='learning rate = 0.001')
    plt.plot(hidden_layer_neural_unit, result_accuracy[3], color='black', marker='d', linestyle='--',
             label='learning rate = 0.0001')
    plt.title("Accuracy for FCNN Using the Scikit-learn Library", fontweight='bold')
    plt.xticks([500, 1000, 1500, 2000])
    plt.xlabel("Numbers of Hidden Layer Neural Unit", fontweight='bold')
    plt.ylabel("Accuracy", fontweight='bold')
    plt.grid(True, linestyle='dashed')  # 添加网格线
    plt.legend(frameon=True, edgecolor='black', bbox_to_anchor=(0.96, 0.6))
    # 设置局部放大区域
    zoom_x = (400, 2100)  # x 轴放大范围
    zoom_y = (0.93, 0.99)  # y 轴放大范围

    # 创建一个新的坐标系，用于绘制局部放大区域
    ax_zoom = plt.axes([0.2, 0.42, 0.42, 0.33])  # 设置新坐标系的位置和大小
    ax_zoom.plot(hidden_layer_neural_unit, result_accuracy[0], color='blue', marker='o', linestyle='--',
                 label='learning rate = 0.1')
    ax_zoom.plot(hidden_layer_neural_unit, result_accuracy[1], color='green', marker='s', linestyle='--',
                 label='learning rate = 0.01')
    ax_zoom.plot(hidden_layer_neural_unit, result_accuracy[2], color='red', marker='^', linestyle='--',
                 label='learning rate = 0.001')
    ax_zoom.plot(hidden_layer_neural_unit, result_accuracy[3], color='black', marker='d', linestyle='--',
                 label='learning rate = 0.0001')
    ax_zoom.set_xlim(zoom_x)  # 设置局部放大区域的 x 轴范围
    ax_zoom.set_ylim(zoom_y)  # 设置局部放大区域的 y 轴范围
    ax_zoom.grid(True, linestyle='dashed')  # 添加网格线

    plt.savefig('figs/fcnn_sklearn.png')
    plt.show()


@get_time
def train_and_predict():
    for j in range(len(learning_rate)):
        print("当前学习率：" + str(learning_rate[j]))
        for i in range(len(hidden_layer_neural_unit)):
            print("当前隐层神经元数量：" + str(hidden_layer_neural_unit[i]))
            model = MLPClassifier(hidden_layer_sizes=hidden_layer_neural_unit[i], activation='relu', solver='adam',
                                  random_state=42, learning_rate='constant', learning_rate_init=learning_rate[j],
                                  max_iter=250)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            result_accuracy[j].append(accuracy)
            print("当前精度：" + str(accuracy))


def fcnn_sklearn():
    train_and_predict()
    draw_fig()
