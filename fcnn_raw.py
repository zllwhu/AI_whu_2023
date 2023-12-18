import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

from utils import get_time


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, max_iter=500, batch_size=200):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.weights1 = None
        self.biases1 = None
        self.weights2 = None
        self.biases2 = None

    def initialize_parameters(self):
        np.random.seed(0)
        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.biases1 = np.zeros((1, self.hidden_size))
        self.weights2 = np.random.randn(self.hidden_size, self.output_size)
        self.biases2 = np.zeros((1, self.output_size))

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def forward_propagation(self, X):
        Z1 = np.dot(X, self.weights1) + self.biases1
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(A1, self.weights2) + self.biases2
        A2 = self.sigmoid(Z2)
        return A1, A2

    def backward_propagation(self, X, y, A1, A2):
        m = X.shape[0]
        dZ2 = A2 - y
        dW2 = np.dot(A1.T, dZ2) / m
        dB2 = np.mean(dZ2, axis=0, keepdims=True)
        dZ1 = np.dot(dZ2, self.weights2.T) * A1 * (1 - A1)
        dW1 = np.dot(X.T, dZ1) / m
        dB1 = np.mean(dZ1, axis=0, keepdims=True)
        return dW1, dB1, dW2, dB2

    def update_parameters(self, dW1, dB1, dW2, dB2):
        self.weights1 -= self.learning_rate * dW1
        self.biases1 -= self.learning_rate * dB1
        self.weights2 -= self.learning_rate * dW2
        self.biases2 -= self.learning_rate * dB2

    def train(self, X, y):
        self.initialize_parameters()
        num_samples = X.shape[0]

        for _ in range(self.max_iter):
            for batch_start in range(0, num_samples, self.batch_size):
                batch_end = batch_start + self.batch_size
                X_batch = X[batch_start:batch_end]
                y_batch = y[batch_start:batch_end]

                A1, A2 = self.forward_propagation(X_batch)
                dW1, dB1, dW2, dB2 = self.backward_propagation(X_batch, y_batch, A1, A2)
                self.update_parameters(dW1, dB1, dW2, dB2)

    def predict(self, X):
        _, A2 = self.forward_propagation(X)
        y_pred = np.argmax(A2, axis=1)
        return y_pred


data_train = np.loadtxt('./pen+based+recognition+of+handwritten+digits/pendigits.tra', delimiter=',')
data_test = np.loadtxt('./pen+based+recognition+of+handwritten+digits/pendigits.tes', delimiter=',')

X_train = data_train[:, :-1]
y_train = data_train[:, -1]
encoder = OneHotEncoder(sparse=False, categories='auto')
y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
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
             markerfacecolor='blue',
             label='learning rate = 0.1')
    plt.plot(hidden_layer_neural_unit, result_accuracy[1], color='green', marker='s', linestyle='--',
             markerfacecolor='green', label='learning rate = 0.01')
    plt.plot(hidden_layer_neural_unit, result_accuracy[2], color='red', marker='^', linestyle='--',
             markerfacecolor='red',
             label='learning rate = 0.001')
    plt.plot(hidden_layer_neural_unit, result_accuracy[3], color='black', marker='d', linestyle='--',
             markerfacecolor='black', label='learning rate = 0.0001')
    plt.title("Accuracy for FCNN Using the Scikit-learn Library", fontweight='bold')
    plt.xticks([500, 1000, 1500, 2000])
    plt.xlabel("Numbers of Hidden Layer Nerual Unit", fontweight='bold')
    plt.ylabel("Accuracy", fontweight='bold')
    plt.grid(True, linestyle='dashed')  # 添加网格线
    plt.legend(frameon=True, edgecolor='black')

    plt.savefig('figs/fcnn_raw.png')
    plt.show()


@get_time
def train_and_predict():
    for j in range(len(learning_rate)):
        print("当前学习率：" + str(learning_rate[j]))
        for i in range(len(hidden_layer_neural_unit)):
            print("当前隐层神经元数量：" + str(hidden_layer_neural_unit[i]))
            output_size = y_train_encoded.shape[1]
            model = NeuralNetwork(X_train.shape[1], hidden_layer_neural_unit[i], output_size,
                                  learning_rate=learning_rate[j], max_iter=2000, batch_size=200)
            model.train(X_train, y_train_encoded)
            y_pred = model.predict(X_test)
            accuracy = np.mean(y_pred == y_test)
            result_accuracy[j].append(accuracy)
            print("当前精度：" + str(accuracy))


def fcnn_raw():
    train_and_predict()
    draw_fig()
