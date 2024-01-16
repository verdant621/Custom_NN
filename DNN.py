import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with random values
        self.weights_input_to_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_to_output = np.random.rand(hidden_size, output_size)
        self.hidden_bias = np.random.rand(hidden_size)
        self.output_bias = np.random.rand(output_size)

    def feedforward(self, X):
        # X is input, shape (num_samples, input_size)
        self.hidden_layer = sigmoid(np.dot(X, self.weights_input_to_hidden) + self.hidden_bias)
        return sigmoid(np.dot(self.hidden_layer, self.weights_hidden_to_output) + self.output_bias)

    def backpropagation(self, X, y_true, y_pred, learning_rate):
        # Calculate the derivative of loss with respect to weights
        error_output_layer = (y_true - y_pred) * sigmoid_derivative(y_pred)
        error_hidden_layer = error_output_layer.dot(self.weights_hidden_to_output.T) * sigmoid_derivative(self.hidden_layer)

        # Calculate the gradients
        gradient_weights_hidden_to_output = self.hidden_layer.T.dot(error_output_layer)
        gradient_weights_input_to_hidden = X.T.dot(error_hidden_layer)

        # Update the weights and biases
        self.weights_hidden_to_output += learning_rate * gradient_weights_hidden_to_output
        self.weights_input_to_hidden += learning_rate * gradient_weights_input_to_hidden
        self.output_bias += learning_rate * error_output_layer.sum(axis=0)
        self.hidden_bias += learning_rate * error_hidden_layer.sum(axis=0)

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            y_pred = self.feedforward(X)
            self.backpropagation(X, y, y_pred, learning_rate)
            if epoch % 10 == 0:
                loss = mse_loss(y, y_pred)
                print("Epoch %d loss: %.3f" % (epoch, loss))

# Example usage:
if __name__ == "__main__":
    # Define dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input for XOR problem
    y = np.array([[0], [1], [1], [0]])  # Output for XOR problem

    # Create a Neural Network with 2 inputs, 5 neurons in hidden layer, and 1 output
    nn = NeuralNetwork(input_size=2, hidden_size=5, output_size=1)

    # Train the neural network
    nn.train(X, y, learning_rate=0.1, epochs=1000)
