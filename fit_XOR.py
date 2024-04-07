import numpy as np
from neural_network import Sequential, LinearLayer, Sigmoid, HyperbolicTangent

# Create the XOR input and output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Define the neural network architecture
network_sigmoid = Sequential()
network_sigmoid.add(LinearLayer(2, 2))
network_sigmoid.add(Sigmoid())
network_sigmoid.add(LinearLayer(2, 1))
network_sigmoid.add(Sigmoid())

network_tanh = Sequential()
network_tanh.add(LinearLayer(2, 2))
network_tanh.add(HyperbolicTangent())
network_tanh.add(LinearLayer(2, 1))
network_tanh.add(HyperbolicTangent())

# Train the network using sigmoid activations
learning_rate_sigmoid = 0.01
num_epochs_sigmoid = 100

print("Sigmoid")
for epoch in range(num_epochs_sigmoid):
    output = network_sigmoid.forward(X)
    loss = np.mean((output - y) ** 2)
    grad_output = 2 * (output - y)
    network_sigmoid.backward(X, grad_output, learning_rate_sigmoid)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss:.4f}")

# Train the network using hyperbolic tangent activations
learning_rate_tanh = 0.1
num_epochs_tanh = 100

print("Hyperbolic tangent")
for epoch in range(num_epochs_tanh):
    output = network_tanh.forward(X)
    loss = np.mean((output - y) ** 2)
    grad_output = 2 * (output - y)
    network_tanh.backward(X, grad_output, learning_rate_tanh)
    
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss:.4f}")


# Verify the XOR problem
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = network_tanh.forward(X_xor)
binary_predictions = (predictions >= 0.5).astype(int)
expected_outputs = np.array([[0], [1], [1], [0]])
accuracy = np.mean(binary_predictions == expected_outputs) * 100

print("Predictions:")
print(binary_predictions)
print("Expected Outputs:")
print(expected_outputs)
print(f"Accuracy: {accuracy:.2f}%")
