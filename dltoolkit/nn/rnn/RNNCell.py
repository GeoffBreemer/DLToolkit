"""Simple RNN cell"""
import numpy as np
from dltoolkit.utils.foundation import softmax
from collections import namedtuple

# Simplify passing around of weights and gradients
Activations = namedtuple('Activations', 'W_aa W_ax W_ay bias_a bias_y')
Gradients = namedtuple('BackpropResults', 'd_x d_a_prev d_W_ax d_W_aa d_b_a d_a0')


class RNNCell:

    def __init__(self, n_x=None, n_a=None, n_y=None, initialize=False):
        """Initialise the cell"""
        self.x = None
        self.a_next = None
        self.a_prev = None

        # Initialize the weight and bias matrices with random numbers if desired, alternatively
        # use set_weights to set them
        if initialize:
            self.W_ax = np.random.randn(n_a, n_x)
            self.W_aa = np.random.randn(n_a, n_a)
            self.W_ay = np.random.randn(n_y, n_a)
            self.bias_a = np.random.randn(n_a, 1)
            self.bias_y = np.random.randn(n_y, 1)

        # Initialise predictions for the current time step
        self.y_pred = None

    def forward_pass(self, x, a_prev):
        """Perform a forward pass for the cell using data for a specific time stamp (x) and the previous cell's
        activations"""
        # Cache the input x for use during backprop
        self.x = x

        # Update the layer's activations for the next time step and predictions for the current time step
        self.a_next = np.tanh(np.dot(self.W_aa, a_prev) + np.dot(self.W_ax, x) + self.bias_a)
        self.y_pred = softmax(np.dot(self.W_ay, self.a_next) + self.bias_y)

        # Current time step's activations become the previous activations for the next time step
        self.a_prev = a_prev

        return self.a_next, self.y_pred, self.a_prev

    def backprop(self, d_a_next):
        """Perform backprop"""
        d_tanh = (1 - np.square(self.a_next)) * d_a_next

        # Compute the gradient of the loss with respect to W_ax
        d_x = np.dot(self.W_ax.T, d_tanh)
        d_W_ax = np.dot(d_tanh, self.x.T)

        # Compute the gradient with respect to W_aa
        d_a_prev = np.dot(self.W_aa.T, d_tanh)
        d_W_aa = np.dot(d_tanh, self.a_prev.T)

        # Compute the gradient with respect to the bias
        d_b_a = np.sum(d_tanh, axis=1, keepdims=True)

        return Gradients(d_x, d_a_prev, d_W_ax, d_W_aa, d_b_a, None)

    def get_activations(self):
        return Activations(self.W_aa, self.W_ax, self.W_ay, self.bias_a, self.bias_y)

    def set_activations(self, Activations):
        self.W_aa = Activations.W_aa
        self.W_ax = Activations.W_ax
        self.W_ay = Activations.W_ay
        self.bias_a = Activations.bias_a
        self.bias_y = Activations.bias_y


if __name__ == '__main__':
    RANDOM_STATE = 1
    NUM_OBSERVATIONS = 10
    NUM_FEATURES = 3
    NUM_OUTPUT = 2
    NUM_HIDDEN = 5

    # Initialize the data and the cell
    np.random.seed(RANDOM_STATE)
    x = np.random.randn(NUM_FEATURES, NUM_OBSERVATIONS)
    a_start = np.random.randn(NUM_HIDDEN, NUM_OBSERVATIONS)
    rnn_cell = RNNCell(NUM_FEATURES, NUM_HIDDEN, NUM_OUTPUT, initialize=True)

    # Perform a forward pass
    print("Forward pass:")
    a_next, y_pred, a_prev = rnn_cell.forward_pass(x, a_start)
    print(a_next[4])
    print()
    print(y_pred[1])
    print()
    print(a_prev[4])

    # Perform backprop
    print("\nBackprop pass:")
    da_next = np.random.randn(5,10)
    gradients = rnn_cell.backprop(da_next)

    print("d_x[1][2] =", gradients.d_x[1][2])
    print("d_x.shape =", gradients.d_x.shape)
    print("d_a_prev[2][3] =", gradients.d_a_prev[2][3])
    print("d_a_prev.shape =", gradients.d_a_prev.shape)
    print("d_W_ax[3][1] =", gradients.d_W_ax[3][1])
    print("d_W_ax.shape =", gradients.d_W_ax.shape)
    print("d_W_aa[1][2] =", gradients.d_W_aa[1][2])
    print("d_W_aa.shape =", gradients.d_W_aa.shape)
    print("d_b_a[4] =", gradients.d_b_a[4])
    print("d_b_a.shape =", gradients.d_b_a.shape)
