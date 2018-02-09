"""Simple RNN cell"""
import numpy as np
from dltoolkit.utils.foundation import softmax


class RNNCell:
    def __init__(self, n_x, n_a, n_y, m):
        # Initialise the activation state for the next and previous time steps
        self.a_next = None
        self.a_prev = np.random.randn(n_a, m)

        # Initialize the weight and bias matrices with random numbers
        self.W_aa = np.random.randn(n_a, n_a)
        self.W_ax = np.random.randn(n_a, n_x)
        self.W_ay = np.random.randn(n_y, n_a)
        self.bias_a = np.random.randn(n_a, 1)
        self.bias_y = np.random.randn(n_y, 1)

        # Initialise the predictions for the current time step
        self.y_pred = None

    def forward_pass(self, xt):
        # Update the layer's activations for the next time step and predictions for the current time step
        self.a_next = np.tanh(np.dot(self.W_aa, self.a_prev) + np.dot(self.W_ax, xt) + self.bias_a)
        self.y_pred = softmax(np.dot(self.W_ay, self.a_next) + self.bias_y)

        # Current time step's activations become the previous activations for the next time step
        self.a_prev = self.a_next

        return self.a_next, self.y_pred


if __name__ == '__main__':
    RANDOM_STATE = 1
    NUM_OBSERVATIONS = 10
    NUM_FEATURES = 3
    NUM_OUTPUT = 2
    NUM_HIDDEN = 5

    # Initialize the data
    np.random.seed(RANDOM_STATE)
    x = np.random.randn(NUM_FEATURES, NUM_OBSERVATIONS)

    # Initialise the RNN
    rnn_cell = RNNCell(NUM_FEATURES, NUM_HIDDEN, NUM_OUTPUT, NUM_OBSERVATIONS)

    # Perform one forward pass
    a_next, y_pred = rnn_cell.forward_pass(x)
    print(a_next[4])
    print(y_pred[1])

    # And another
    a_next, y_pred = rnn_cell.forward_pass(x)
    print(a_next[4])
    print(y_pred[1])

    print("======")
    print(y_pred)
    print(y_pred.shape)
