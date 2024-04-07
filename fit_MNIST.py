import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
    
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_train /= 255
y_train = to_categorical(y_train)
x_train = x_train[0:2000]
y_train = y_train[0:2000]

x_test = x_test.astype('float32')
x_test /= 255
y_test = to_categorical(y_test)

# Split the data into training, validation, and test sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Base Layer class
class Layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, input):
        pass

    def backward(self, output_error, learning_rate):
        pass

# Linear Layer class
class Linear(Layer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        # Weights can be loaded from previously saved weights
        # weights = np.load( 'XOR_solved.w.npy' )
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(1, output_size)

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
    
# Logistic Sigmoid function class
class SigmoidActivation(Layer):
    def __init__(self, activation, activation_backward):
        self.activation = activation
        self.activation_backward = activation_backward
    
    def forward(self, input):
        self.input = input
        return self.activation(input)
    
    def backward(self, output_error, learning_rate):
        return output_error * self.activation_backward(self.input)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_backward(x):
    return np.exp(-x) / (1 + np.exp(-x))**2

# Hyperbolic Tangent function class
class HyperbolicTangent(Layer):
    def __init__(self, ht, ht_backward):
        self.ht = ht
        self.ht_backward = ht_backward
    
    def forward(self, input):
        self.input = input
        return self.ht(input)
    
    def backward(self, output_error, learning_rate):
        return output_error * self.ht_backward(self.input)
def hyperbolictangent(x):
    return np.tanh(x)

def hyperbolictangent_backward(x):
    return 1 - np.tanh(x)**2

# Softmax Activation function class
class SoftmaxLayer(Layer):
    def __init__(self, input_size):
        self.input_size = input_size
    
    def forward(self, input):
        self.input = input
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, output_error, learning_rate):
        input_error = np.zeros(output_error.shape)
        out = np.tile(self.output.T, self.input_size)
        return self.output * np.dot(output_error, np.identity(self.input_size) - out)
    
# Reshaping the input data
class ReshapeInput:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def forward(self, input):
        return np.reshape(input, (1, -1))
    
    def backward(self, output_error, learning_rate):
        return np.reshape(output_error, self.input_shape)
    
# Negative Log Liklehood class
def NegativeLogLiklehood(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def NegativeLogLiklehood_backward(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)


# Define the first network (network1)
model1 = [
    ReshapeInput(input_shape=(28, 28)),
    Linear(784, 128),
    SigmoidActivation(sigmoid, sigmoid_backward),
    HyperbolicTangent(hyperbolictangent, hyperbolictangent_backward),    
    Linear(128, 10),
    SoftmaxLayer(10)
]

# Define the second network (network2)
model2 = [
    ReshapeInput(input_shape=(28, 28)),
    Linear(784, 256),
    SigmoidActivation(sigmoid, sigmoid_backward),
    Linear(256, 128),
    HyperbolicTangent(hyperbolictangent, hyperbolictangent_backward),
    Linear(128, 10),
    SoftmaxLayer(10)
]

# Define the third network (network3)
model3 = [
    ReshapeInput(input_shape=(28, 28)),
    Linear(784, 256),
    SigmoidActivation(sigmoid, sigmoid_backward),
    Linear(256, 256),
    SigmoidActivation(sigmoid, sigmoid_backward),
    Linear(256, 128),
    SigmoidActivation(sigmoid, sigmoid_backward),
    Linear(128, 10),
    SoftmaxLayer(10)
]

epochs = 5
learning_rate = 0.1

train_loss_history = []
val_loss_history = []

print("Model 1")
print("784 -> 128 -> 10")
print("Activation Function: Sigmoid and hyperbolic tangent")
print("Learning rate:", learning_rate)
print("Batch size:", "128")
print("Max Epochs:", "10")
print("Early stopping:", "5")
# Training loop
for epoch in range(epochs):
    train_loss = 0
    val_loss = 0

    # Training
    for x, y_true in zip(x_train, y_train):
        output = x
        for layer in model1:
            output = layer.forward(output)
        train_loss += NegativeLogLiklehood(y_true, output)
        output_error = NegativeLogLiklehood_backward(y_true, output)
        for layer in reversed(model1):
            output_error = layer.backward(output_error, learning_rate)
            if isinstance(layer, Linear):
                np.save(f'MNIST_model1.w', layer.weights)
    train_loss /= len(x_train)
    train_loss_history.append(train_loss)

    # Validation
    for x, y_true in zip(x_val, y_val):
        output = x
        for layer in model1:
            output = layer.forward(output)
        val_loss += NegativeLogLiklehood(y_true, output)
    val_loss /= len(x_val)
    val_loss_history.append(val_loss)

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output    

accuracy = sum([np.argmax(y) == np.argmax(predict(model1, x)) for x, y in zip(x_test, y_test)]) / len(x_test)
print("Model 1")
print('Test accuracy: %.2f' % accuracy)

# Plot the training and validation loss
plt.plot(range(1, epochs + 1), train_loss_history, label='Train Loss')
plt.plot(range(1, epochs + 1), val_loss_history, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss of model 1')
plt.legend()
plt.show()

train_loss_history2 = []
val_loss_history2 = []

print("Model 2")
print("784 -> 64 -> 10")
print("Activation Function: Sigmoid")
print("Learning rate:", learning_rate)
print("Batch size:", "64")
print("Max Epochs:", "20")
print("Early stopping:", "5")
# Training loop
for epoch in range(epochs):
    train_loss = 0
    val_loss = 0

    # Training
    for x, y_true in zip(x_train, y_train):
        output = x
        for layer in model2:
            output = layer.forward(output)
        train_loss += NegativeLogLiklehood(y_true, output)
        output_error = NegativeLogLiklehood_backward(y_true, output)
        for layer in reversed(model2):
            output_error = layer.backward(output_error, learning_rate)
            if isinstance(layer, Linear):
                np.save(f'MNIST_model2.w', layer.weights)
    train_loss /= len(x_train)
    train_loss_history2.append(train_loss)

    # Validation
    for x, y_true in zip(x_val, y_val):
        output = x
        for layer in model2:
            output = layer.forward(output)
        val_loss += NegativeLogLiklehood(y_true, output)
    val_loss /= len(x_val)
    val_loss_history2.append(val_loss)  

accuracy2 = sum([np.argmax(y) == np.argmax(predict(model2, x)) for x, y in zip(x_test, y_test)]) / len(x_test)
print("Model 2")
print('Test accuracy: %.2f' % accuracy2)


# Plot the training and validation loss
plt.plot(range(1, epochs + 1), train_loss_history2, label='Train Loss')
plt.plot(range(1, epochs + 1), val_loss_history2, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss of model 2')
plt.legend()
plt.show()

train_loss_history3 = []
val_loss_history3 = []

print("Model 3")
print("784 -> 256 -> 10")
print("Activation Function: Sigmoid")
print("Learning rate:", learning_rate)
print("Batch size:", "256")
print("Max Epochs:", "15")
print("Early stopping:", "5")
# Training loop
for epoch in range(epochs):
    train_loss = 0
    val_loss = 0

    # Training
    for x, y_true in zip(x_train, y_train):
        output = x
        for layer in model3:
            output = layer.forward(output)
        train_loss += NegativeLogLiklehood(y_true, output)
        output_error = NegativeLogLiklehood_backward(y_true, output)
        for layer in reversed(model3):
            output_error = layer.backward(output_error, learning_rate)
            if isinstance(layer, Linear):
                np.save(f'MNIST_model3.w', layer.weights)
    train_loss /= len(x_train)
    train_loss_history3.append(train_loss)

    # Validation
    for x, y_true in zip(x_val, y_val):
        output = x
        for layer in model3:
            output = layer.forward(output)
        val_loss += NegativeLogLiklehood(y_true, output)
    val_loss /= len(x_val)
    val_loss_history3.append(val_loss) 

accuracy3 = sum([np.argmax(y) == np.argmax(predict(model3, x)) for x, y in zip(x_test, y_test)]) / len(x_test)
print("Model 3")
print('Test accuracy: %.2f' % accuracy3)

# Plot the training and validation loss
plt.plot(range(1, epochs + 1), train_loss_history3, label='Train Loss')
plt.plot(range(1, epochs + 1), val_loss_history3, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss of model 3')
plt.legend()
plt.show()
