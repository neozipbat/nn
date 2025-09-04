import numpy as np
from model import NeuralNetwork, load_mnist

test_x, test_y = load_mnist("mnist_test.csv")

input_size = 784
hidden_size = 200
output_size = 10
lr = 0.1

nn = NeuralNetwork(input_size, hidden_size, output_size, lr)

# بارگذاری وزن‌های آموزش داده شده
nn.w_i_h = np.load("w_i_h.npy")
nn.w_h_o = np.load("w_h_o.npy")

correct = 0
for x, y in zip(test_x, test_y):
    if nn.predict(x) == y:
        correct += 1

accuracy = correct / len(test_y)
print(f"🎯 دقت روی دیتاست تست: {accuracy:.4f}")
