# Neural Network Implementation

This project implements a simple neural network using Python and NumPy.

## Overview

The neural network implemented here is a basic feedforward neural network designed for classification tasks. It consists of an input layer, one hidden layer, and an output layer. The hidden layer uses the ReLU (Rectified Linear Unit) activation function, while the output layer employs the softmax activation function for multi-class classification.

## Dependencies

- Python 3.x
- NumPy
- pandas

## Usage

### Training the Neural Network

To train the neural network, you can follow these steps:

1. Prepare your training data: Make sure your training data is properly formatted. In this example, the MNIST dataset is used, and it's assumed that the data is stored in CSV files (`mnist_train.csv` and `mnist_test.csv`).

2. Import the `Network` class from the `NN` module:

    ```python
    from NN import Network
    ```

3. Instantiate a `Network` object:

    ```python
    network = Network()
    ```

4. Fit the model to your training data:

    ```python
    network.fit(X_train, y_train)
    ```

### Making Predictions

After training the neural network, you can use it to make predictions:

1. Prepare your test data: Ensure your test data is properly formatted.

2. Use the `predict` method of the `Network` object to make predictions:

    ```python
    predictions = network.predict(X_test)
    ```

3. Evaluate the performance of your model:

    ```python
    accuracy = network.get_accuracy(predictions, y_test)
    ```

## Files

- `NN.py`: Contains the implementation of the `Network` class, which defines the neural network model.
- `main.py`: A sample script demonstrating how to train the neural network using the MNIST dataset.

## Contributing

Contributions to this project are welcome. Feel free to submit issues or pull requests.

