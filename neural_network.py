import numpy as np
import scipy.special


class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.inputs = input_nodes
        self.hidden = hidden_nodes
        self.output = output_nodes
        self.lr = learning_rate

        self.wih = np.random.normal(0.0, pow(self.hidden, -0.5), (self.hidden, self.inputs))
        self.who = np.random.normal(0.0, pow(self.output, -0.5), (self.output, self.hidden))

        self.activation_function = lambda x: scipy.special.expit(x)

    # Train the network using back-propagation of errors
    def train(self, inputs_list, targets_list):
        inputs_array = np.array(inputs_list, ndmin=2).T
        targets_array = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs_array)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets_array - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs_array))

    # Query the network
    def query(self, inputs_list):
        inputs_array = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs_array)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs