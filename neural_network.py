import numpy as np

class Layer:

    def forward(self, input):
        raise NotImplementedError

    def backward(self, input, grad_output):
        raise NotImplementedError

    def save_weights(self, filepath):
        pass

    def load_weights(self, filepath):
        pass


class LinearLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros(output_size)

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, input, grad_output, learning_rate):
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0)

        self.weights -= learning_rate * grad_weights  # Remove the transpose operation

        # Saving weights 
        np.save('XOR_solved.w.npy', self.weights)
        self.bias -= learning_rate * grad_bias

        return grad_input

class Sigmoid(Layer):
    def forward(self, x):
        self.output = 1.0 / (1 + np.exp(-x))
        return self.output

    def backward(self, input, grad_output, learning_rate):
        return self.output * (1 - self.output) * grad_output

class HyperbolicTangent(Layer):
    def forward(self, x):
        self.output = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return self.output

    def backward(self, input, grad_output, learning_rate):
        return (1 - self.output ** 2) * grad_output


class Softmax(Layer):
    def forward(self, x):
        x_shifted = x - np.max(x)
        x_exp = np.exp(x_shifted)
        a = x_exp / x_exp.sum(axis=-1, keepdims=True)
        return a

    def backward(self, input, grad_output, learning_rate=None, batch_size=None):
        return grad_output


class CrossEntropyLoss(Layer):
    def forward(self, input, target):
        self.target = np.eye(10)[target]  # Convert target to one-hot encoded format
        epsilon = 1e-7
        self.prediction = np.clip(input, epsilon, 1 - epsilon)
        loss = -np.sum(self.target * np.log(self.prediction + epsilon)) / self.target.shape[0]
        return loss

    def backward(self, input, grad_output):
        return (self.prediction - self.target) / input.shape[0]


class Sequential(Layer):
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        self.layer_outputs = []
        output = input
        for layer in self.layers:
            output = layer.forward(output)
            self.layer_outputs.append(output)
        return output

    def backward(self, input, grad_output, learning_rate):
        num_layers = len(self.layers)
        for i in range(num_layers - 1, -1, -1):
            grad_output = self.layers[i].backward(self.layer_outputs[i], grad_output, learning_rate)
        return grad_output

    def fit(self, X_train, y_train, X_val, y_val, loss_fn, learning_rate, batch_size, max_epochs, early_stopping):
        train_loss_history = []
        val_loss_history = []
        best_val_loss = float('inf')
        steps_without_improvement = 0
        
        for epoch in range(max_epochs):
            epoch_loss = 0
            num_batches = len(X_train) // batch_size
            
            for i in range(num_batches):
                batch_start = i * batch_size
                batch_end = (i + 1) * batch_size

                # Forward pass
                batch_X = X_train[batch_start:batch_end]
                batch_y = y_train[batch_start:batch_end]
                predictions = self.forward(batch_X)
                
                # Calculate loss and gradient
                loss = loss_fn.forward(predictions, batch_y)
                grad_output = loss_fn.backward(predictions, batch_y)
                epoch_loss += loss
                
                # Backward pass
                self.backward(batch_X, grad_output, learning_rate)
            
            # Calculate average loss for the epoch
            epoch_loss /= num_batches
            train_loss_history.append(epoch_loss)
            
            # Evaluate on the validation set
            val_predictions = self.forward(X_val)
            val_loss = loss_fn.forward(val_predictions, y_val)
            val_loss_history.append(val_loss)

             # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
                if steps_without_improvement >= early_stopping:
                    break
        
        return train_loss_history, val_loss_history
