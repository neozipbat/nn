import numpy as np
from model import NeuralNetwork, load_mnist

train_x, train_y = load_mnist("mnist_train.csv")

input_size = 784
hidden_size = 200
output_size = 10
lr = 0.1
nn = NeuralNetwork(input_size, hidden_size, output_size, lr)

epochs = 5
for e in range(epochs):
    for x, y in zip(train_x, train_y):
        nn.train(x, y)
    print(f"Epoch {e+1}/{epochs} تمام شد ✅")

# ذخیره مدل با np.save
np.save("w_i_h.npy", nn.w_i_h)
np.save("w_h_o.npy", nn.w_h_o)
print("✅ مدل ذخیره شد")
