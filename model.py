import numpy as np
from scipy.special import expit, logit

def load_mnist(file_path):
    data = np.loadtxt(file_path, delimiter=",")
    labels = data[:, 0].astype(int)
    features = data[:, 1:] / 255.0 * 0.98 + 0.01
    return features, labels

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate

        self.w_i_h = np.random.normal(0, pow(self.input_size, -0.5), (self.hidden_size, self.input_size))
        self.w_h_o = np.random.normal(0, pow(self.hidden_size, -0.5), (self.output_size, self.hidden_size))

        self.activation = lambda x: expit(x)
        self.inverse_activation = lambda x: logit(x)

    def query(self, inputs):
        inputs = np.array(inputs, ndmin=2).T
        hidden_outputs = self.activation(np.dot(self.w_i_h, inputs))
        final_outputs = self.activation(np.dot(self.w_h_o, hidden_outputs))
        return final_outputs

    def train(self, inputs, target):
        inputs = np.array(inputs, ndmin=2).T
        targets = np.zeros((self.output_size, 1)) + 0.01
        targets[target, 0] = 0.99

        hidden_outputs = self.activation(np.dot(self.w_i_h, inputs))
        final_outputs = self.activation(np.dot(self.w_h_o, hidden_outputs))

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.w_h_o.T, output_errors)

        self.w_h_o += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)), hidden_outputs.T)
        self.w_i_h += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), inputs.T)

    def predict(self, inputs):
        return np.argmax(self.query(inputs))
    
    def reverse_query(self, target):
        t = np.zeros(self.output_size) + 0.01
        t[target] = 0.99
        t = self.inverse_activation(t)
        h = np.dot(self.w_h_o.T, t)
        h -= np.min(h)
        h /= np.max(h)
        h = h * 0.98 + 0.01
        i = self.inverse_activation(h)
        i = np.dot(self.w_i_h.T, i)
        i -= np.min(i)
        i /= np.max(i)
        i = i * 0.98 + 0.01
        return i
