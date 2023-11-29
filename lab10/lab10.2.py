import numpy as np

class Neuron:
    def __init__(self, input_size):
        self._weights = np.random.uniform(size=(input_size,))
        self._bias = np.random.uniform()

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        self._input = x
        self._activation = np.dot(x, self._weights) + self._bias
        self._output = self._sigmoid(self._activation)
        return self._output

    def backward(self, loss):
        delta = loss * self._sigmoid_derivative(self._output)
        self._weights += self._input * delta
        self._bias += delta

class Model:
    def __init__(self, input_size, hidden_size, output_size):
        self.neuron_hidden1 = Neuron(input_size)
        self.neuron_hidden2 = Neuron(input_size)
        self.neuron_output = Neuron(hidden_size)

    def forward(self, x):
        out_hidden1 = self.neuron_hidden1.forward(x)
        out_hidden2 = self.neuron_hidden2.forward(x)
        combined_output = np.array([out_hidden1, out_hidden2])
        output = self.neuron_output.forward(combined_output.flatten())  # исправлено
        return output

    def backward(self, x, loss):
        loss_hidden1 = loss * self.neuron_output._weights[0] * self.neuron_hidden1._sigmoid_derivative(self.neuron_hidden1._output)
        loss_hidden2 = loss * self.neuron_output._weights[1] * self.neuron_hidden2._sigmoid_derivative(self.neuron_hidden2._output)
        self.neuron_hidden1.backward(loss_hidden1)
        self.neuron_hidden2.backward(loss_hidden2)
        self.neuron_output.backward(loss)

if __name__ == "__main__":
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    expected_output = np.array([[0], [1], [1], [0]])

    input_size = 2
    hidden_size = 2
    output_size = 1
    model = Model(input_size, hidden_size, output_size)

    # Training algorithm
    epochs = 10000
    lr = 0.1
    for _ in range(epochs):
        for i in range(len(inputs)):
            x = inputs[i]
            y = model.forward(x)
            err = expected_output[i] - y
            model.backward(x, err)

    print("\nOutput from neural network after 10,000 epochs: ", end='')
    for i in range(len(inputs)):
        x = inputs[i]
        y = model.forward(x)
        print(y, end=' ')
