import numpy as np

# Define the sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(s):
    return s * (1 - s)

# A RNN with a single hidden layer
class SimpleRNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Weights and biases initialization
        self.Wxh = np.random.randn(hidden_dim, input_dim) * np.sqrt(1. / input_dim)
        self.Whh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(1. / hidden_dim)
        self.Why = np.random.randn(output_dim, hidden_dim) * np.sqrt(1. / hidden_dim)
        self.bh = np.zeros((hidden_dim, 1))
        self.by = np.zeros((output_dim, 1))

    def forward(self, inputs):
        h_prev = np.zeros((self.Wxh.shape[0], 1))
        self.hs, self.xs, self.ys = {}, {}, {}
        self.hs[-1] = np.copy(h_prev)

        # Store inputs and outputs for each time step
        for t in range(len(inputs)):
            self.xs[t] = inputs[t]
            h_raw = np.dot(self.Wxh, self.xs[t]) + np.dot(self.Whh, h_prev) + self.bh
            h_prev = np.tanh(h_raw)
            y_raw = np.dot(self.Why, h_prev) + self.by
            self.ys[t] = sigmoid(y_raw)
            self.hs[t] = h_prev

        return self.ys, self.hs

    def backward(self, inputs, true_outputs, learning_rate=0.01):
        dWhh, dWxh, dWhy = np.zeros_like(self.Whh), np.zeros_like(self.Wxh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(self.hs[0])

        for t in reversed(range(len(inputs))):
            dy = self.ys[t] - true_outputs[t]
            dWhy += np.dot(dy, self.hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext
            dhraw = (1 - self.hs[t] ** 2) * dh
            dbh += dhraw
            dWxh += np.dot(dhraw, self.xs[t].T)
            dWhh += np.dot(dhraw, self.hs[t-1].T)
            dhnext = np.dot(self.Whh.T, dhraw)

        # Clip gradients to prevent exploding gradients
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        # Update parameters with gradient descent
        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby

rnn = SimpleRNN(input_dim=2, hidden_dim=3, output_dim=1)

# Example input and output sequence (here input_dim=2 and output_dim=1)
inputs = [np.random.rand(2, 1) for _ in range(5)] # input sequence
true_outputs = [np.random.rand(1, 1) for _ in range(5)] # true output sequence

# Forward pass
ys, hs = rnn.forward(inputs)

# Backward pass and optimization
rnn.backward(inputs, true_outputs)

# Check final outputs
ys, hs = rnn.forward(inputs)
print("Final outputs after single backpropagation step:", ys)
