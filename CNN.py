import numpy as np

def convolve2d(X, W, b, stride=1, padding=0):
    n_filters, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    
    h_out = (h_x - h_filter + 2 * padding) // stride + 1
    w_out = (w_x - w_filter + 2 * padding) // stride + 1
    
    # Add padding to the input image if required
    if padding != 0:
        X_padded = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant')
    else:
        X_padded = X

    Z = np.zeros((n_x, n_filters, h_out, w_out))
    
    for i in range(n_x):  # loop over the batch of images
        for f in range(n_filters):  # loop over the filters
            for h in range(h_out):  # loop over the height of the output
                for w in range(w_out):  # loop over the width of the output
                    h_start = h * stride
                    h_end = h_start + h_filter
                    w_start = w * stride
                    w_end = w_start + w_filter
                    
                    # Element-wise multiplication and summation
                    Z[i, f, h, w] = np.sum(X_padded[i, :, h_start:h_end, w_start:w_end] * W[f]) + b[f]
    
    Z = Z.reshape(n_x, -1)  # Flatten the output
    return Z

def relu(Z):
    return np.maximum(0, Z)

def maxpool(X, size=2, stride=2):
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - size) // stride + 1
    w_out = (w_x - size) // stride + 1
    
    M = np.zeros((n_x, d_x, h_out, w_out))

    for i in range(n_x):
        for d in range(d_x):
            for h in range(h_out):
                for w in range(w_out):
                    h_start = h * stride
                    h_end = h_start + size
                    w_start = w * stride
                    w_end = w_start + size
                    
                    # Taking the max over the pooling window
                    M[i, d, h, w] = np.max(X[i, d, h_start:h_end, w_start:w_end])
    
    M = M.reshape(n_x, -1)  # Flatten the output
    return M

class SimpleCNN:
    def __init__(self, num_filters, filter_size, input_dim, num_classes):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Initialize weights and biases for the convolutional layer
        self.W_conv = np.random.randn(num_filters, input_dim[0], filter_size, filter_size) * 0.01
        self.b_conv = np.zeros((num_filters, 1))
        
        # Calculate dimensions after the conv and pool layers to initialize FC weights
        self.flat_size = num_filters * ((input_dim[1] - filter_size + 1) // 2) * ((input_dim[2] - filter_size + 1) // 2)
        self.W_fc = np.random.randn(self.flat_size, num_classes) * 0.01
        self.b_fc = np.zeros((num_classes, 1))
    
    def forward(self, X):
        # Convolutional layer - forward pass
        Z_conv = convolve2d(X, self.W_conv, self.b_conv)
        A_conv = relu(Z_conv)
        
        # Pooling layer
        A_pool = maxpool(A_conv.reshape(X.shape[0], self.num_filters, X.shape[2] - self.filter_size + 1, X.shape[3] - self.filter_size + 1))
        
        # Fully connected layer
        Z_fc = np.dot(A_pool, self.W_fc) + self.b_fc.T
        
        # Softmax layer can be added here after implementing backpropagation for training
        
        return Z_fc  # For now, we'll just return unnormalized scores

# Parameters for the CNN
num_filters = 8
filter_size = 3
input_dim = (1, 28, 28)  # Example for MNIST (grayscale image)
num_classes = 10

# Instantiate the CNN
cnn = SimpleCNN(num_filters, filter_size, input_dim, num_classes)

# Mock input data
X = np.random.randn(5, *input_dim)  # A batch of 5 MNIST-like images

# Forward 
scores = cnn.forward(X)

# Display the shape of the output from the CNN
print("Output shape from the CNN:", scores.shape)

# Remember that this code only shows the forward pass, without any training capability through backpropagation.
