from neural_network import NeuralNetwork
import pandas as pd
import numpy as np

MNIST_train = pd.read_csv('data/MNIST/mnist_train_100.csv', header=None)
MNIST_test = pd.read_csv('data/MNIST/mnist_test_10.csv', header=None)

MNIST_fashion_train = pd.read_csv('data/Fashion_MNIST/fashion_mnist_train_1000.csv', header=None)
MNIST_fashion_test = pd.read_csv('data/Fashion_MNIST/fashion_mnist_test_10.csv', header=None)

learning_rate = 0.15  # 0.08 for MNIST Fashion dataset
hidden_nodes = 1000
epoch = 10  # 3 for MNIST Fashion dataset

neural_network = NeuralNetwork(784, hidden_nodes, 10, learning_rate)

print('Training MLP Neural Network')
for y in range(epoch):
    print('Epoch:', y+1)

    correct = 0

    for x in range(MNIST_train.shape[0]):
        input_list = MNIST_train.loc[x].tolist()
        inputs = (np.asarray(input_list[1:]) / 255.0 * 0.99) + 0.01

        target_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        target_list[input_list[0]] = 1

        neural_network.train(inputs, target_list)

    for x in range(MNIST_test.shape[0]):
        input_list = MNIST_test.loc[x].tolist()
        inputs = (np.asarray(input_list[1:]) / 255.0 * 0.99) + 0.01

        target = input_list[0]
        predicted = np.argmax(neural_network.query(inputs))

        if target == predicted:
            correct += 1

    print('Accuracy after epoch', y+1, 'is', correct/MNIST_test.shape[0])

print('final accuracy:', correct/MNIST_test.shape[0])




