from dltoolkit.utils.foundation import *
from dltoolkit.utils.utils_rnn import *
from dltoolkit.nn.rnn import RNNCell


class SimpleRNN:
    def __init__(self, n_time_step, n_x, n_a, n_y, m):
        self.n_time_step = n_time_step

        # Initialize the activations and predictions
        self.a = np.zeros((n_a, m, n_time_step))
        self.y_pred = np.zeros((n_y, m, n_time_step))

        # Initialize the RNN cell
        self.rnn_cell = RNNCell(n_x, n_a, n_y, m)

    def forward_pass(self, xt, a_start=None):
        # Overwrite initial activations if required
        if a_start is not None:
            self.rnn_cell.a_prev = a_start

        # Loop over all time steps using the activations of the previous cell and passing the current cell
        # the input for the current time step
        for t in range(self.n_time_step):
            self.a[:, :, t], self.y_pred[:, :, t] = self.rnn_cell.forward_pass(xt[:, :, t])

        # Return activations and predictions
        return self.a, self.y_pred


if __name__ == '__main__':
    RANDOM_STATE = 1
    NUM_OBSERVATIONS = 10
    NUM_FEATURES = 3
    NUM_OUTPUT = 2
    NUM_HIDDEN = 5
    NUM_TIMESTEPS = 4

    # Initialize the data
    np.random.seed(RANDOM_STATE)
    x = np.random.randn(NUM_FEATURES, NUM_OBSERVATIONS, NUM_TIMESTEPS)

    # Initialise the SimpleRNN
    rnn = SimpleRNN(NUM_TIMESTEPS, NUM_FEATURES, NUM_HIDDEN, NUM_OUTPUT, NUM_OBSERVATIONS)

    # Perform one forward pass
    a_next, y_pred = rnn.forward_pass(x)
    print("a[4][1] = ", a_next[4][1])
    print("a.shape = ", a_next.shape)
    print("y_pred[1][3] =", y_pred[1][3])
    print("y_pred.shape = ", y_pred.shape)

    # And another
    # a_next, y_pred = rnn.forward_pass(x, a_next[:,:,-1])
    a_next, y_pred = rnn.forward_pass(x)
    print("a[4][1] = ", a_next[4][1])
    print("a.shape = ", a_next.shape)
    print("y_pred[1][3] =", y_pred[1][3])
    print("y_pred.shape = ", y_pred.shape)
