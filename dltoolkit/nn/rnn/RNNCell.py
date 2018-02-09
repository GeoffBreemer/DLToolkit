"""Simple RNN cell"""
import numpy as np
from dltoolkit.utils.foundation import softmax


class RNNCell:
    def __init__(self, n_x, n_a, n_y, m):
        # Initialise the activation state for the next and previous time steps
        self.a_next = None
        self.a_prev = np.random.randn(n_a, m)

        # Initialize the weight and bias matrices with random numbers
        # self.W_aa = np.random.randn(n_a, n_a)       # TODO: for forward pass test
        # self.W_ax = np.random.randn(n_a, n_x)       # TODO: forward pass test
        self.W_ax = np.random.randn(n_a, n_x)       # TODO: for backprop test
        self.W_aa = np.random.randn(n_a, n_a)       # TODO: for backprop test
        self.W_ay = np.random.randn(n_y, n_a)
        self.bias_a = np.random.randn(n_a, 1)
        self.bias_y = np.random.randn(n_y, 1)

        # Initialise the predictions for the current time step
        self.y_pred = None

        # Current and previous activations, input, weights and biases are cached during the forward pass for use during
        # backpropagation
        self.cache = None

    def forward_pass(self, xt):
        # Cache for use during backprop
        self.xt = xt

        # Update the layer's activations for the next time step and predictions for the current time step
        self.a_next = np.tanh(np.dot(self.W_aa, self.a_prev) + np.dot(self.W_ax, xt) + self.bias_a)
        self.y_pred = softmax(np.dot(self.W_ay, self.a_next) + self.bias_y)

        # Current time step's activations become the previous activations for the next time step
        self.a_prev = self.a_next

        return self.a_next, self.y_pred

    def backprop(self, d_a_next):
        print("self.a_prev.T")
        print(self.a_prev.T)
        print()

        d_tanh = (1 - np.square(self.a_next)) * d_a_next

        # compute the gradient of the loss with respect to W_ax
        d_xt = np.dot(self.W_ax.T, d_tanh)
        d_W_ax = np.dot(d_tanh, self.xt.T)

        # compute the gradient with respect to W_aa
        d_a_prev = np.dot(self.W_aa.T, d_tanh)
        d_W_aa = np.dot(d_tanh, self.a_prev.T)

        print("cell d_tanh")
        print(d_tanh)
        print()

        print("cell d_W_aa")
        print(d_W_aa)
        print()

        # compute the gradient with respect to b
        d_b_a = np.sum(d_tanh, axis=1, keepdims=True)

        # Store the gradients in a python dictionary
        self.gradients = {"d_x": d_xt, "d_a_prev": d_a_prev, "d_W_ax": d_W_ax, "d_W_aa": d_W_aa, "d_b_a": d_b_a}

        return self.gradients

if __name__ == '__main__':
    RANDOM_STATE = 1
    NUM_OBSERVATIONS = 10
    NUM_FEATURES = 3
    NUM_OUTPUT = 2
    NUM_HIDDEN = 5

    # Initialize the data and the cell
    np.random.seed(RANDOM_STATE)
    x = np.random.randn(NUM_FEATURES, NUM_OBSERVATIONS)
    rnn_cell = RNNCell(NUM_FEATURES, NUM_HIDDEN, NUM_OUTPUT, NUM_OBSERVATIONS)

    # Perform one forward pass
    a_next, y_pred = rnn_cell.forward_pass(x)
    print(a_next[4])
    print()
    print(y_pred[1])

    # Perform backprop
    da_next = np.random.randn(5,10)
    print("============")
    print(da_next)
    exit()
    gradients = rnn_cell.backprop(da_next)

    print("\nd_xt[1][2] =", gradients["d_x"][1][2])
    print("d_x.shape =", gradients["d_x"].shape)
    print("d_a_prev[2][3] =", gradients["d_a_prev"][2][3])
    print("d_a_prev.shape =", gradients["d_a_prev"].shape)
    print("d_W_ax[3][1] =", gradients["d_W_ax"][3][1])
    print("d_W_ax.shape =", gradients["d_W_ax"].shape)
    print("d_W_aa[1][2] =", gradients["d_W_aa"][1][2])
    print("d_W_aa.shape =", gradients["d_W_aa"].shape)
    print("d_b_a[4] =", gradients["d_b_a"][4])
    print("d_b_a.shape =", gradients["d_b_a"].shape)
