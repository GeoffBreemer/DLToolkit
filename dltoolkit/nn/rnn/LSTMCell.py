"""Simple LSTM cell"""
import numpy as np
from dltoolkit.utils.foundation import sigmoid, softmax


class LSTMCell():
    def __init__(self, n_x, n_a, n_y, m):
        # Initialise the activation state for the next and previous time steps
        self.a_next = None
        self.c_next = None
        self.a_prev = np.random.randn(n_a, m)
        self.c_prev = np.zeros((n_a, m))

        # Initialize the gate and cell value weight and bias matrices with random numbers
        self.W_f = np.random.randn(n_a, n_a + n_x)
        self.bias_f = np.random.randn(n_a, 1)
        self.W_i = np.random.randn(n_a, n_a + n_x)
        self.bias_i = np.random.randn(n_a, 1)
        self.W_o = np.random.randn(n_a, n_a + n_x)
        self.bias_o = np.random.randn(n_a, 1)
        self.W_c = np.random.randn(n_a, n_a + n_x)
        self.bias_c = np.random.randn(n_a, 1)
        self.W_y = np.random.randn(n_y, n_a)
        self.bias_y = np.random.randn(n_y, 1)

        # Initialise the predictions for the current time step
        self.y_pred = None

        # Concatenate a_prev and xt
        self.concat = np.empty((n_a + n_x, m))
        self.concat[: n_a, : ] = self.a_prev

    def forward_pass(self, xt):
        # Concatenate a_prev and xt
        self.concat[self.a_prev.shape[0]:, :] = xt

        # Calculate gate and candidate value(s)
        gate_forget = sigmoid(np.dot(self.W_f, self.concat) + self.bias_f)      # forget gate
        gate_filter = sigmoid(np.dot(self.W_i, self.concat) + self.bias_i)      # filter gate
        c_cand = np.tanh(np.dot(self.W_c, self.concat) + self.bias_c)           # candidate cell value
        self.c_next = gate_forget * self.c_prev + gate_filter * c_cand          # new cell value
        gate_output = sigmoid(np.dot(self.W_o, self.concat) + self.bias_o)      # output gate

        # Update the layer's activations for the next time step and predictions for the current time step
        self.a_next = gate_output * np.tanh(self.c_next)
        self.y_pred = softmax(np.dot(self.W_y, self.a_next) + self.bias_y)

        # Current time step's activations and candidate values become the previous activations and candidate
        # values for the next time step
        self.a_prev = self.a_next
        self.c_prev = self.c_next

        # Update the concatenated matrix [a_(t-1), xt] with the updated activations
        self.concat[:self.a_prev.shape[0], : ] = self.a_next

        return self.a_next, self.y_pred, self.c_next


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
    lstm_cell = LSTMCell(NUM_FEATURES, NUM_HIDDEN, NUM_OUTPUT, NUM_OBSERVATIONS)

    # Perform one forward pass
    a_next, y_pred, c_next, = lstm_cell.forward_pass(x)
    print(a_next[4])
    print(c_next[2])
    print(y_pred[1])

    # And a second pass
    print()
    a_next, y_pred, c_next, = lstm_cell.forward_pass(x)
    print(a_next[4])
    print(c_next[2])
    print(y_pred[1])
