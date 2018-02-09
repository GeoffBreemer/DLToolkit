from dltoolkit.utils.foundation import *
from dltoolkit.utils.utils_rnn import *
from dltoolkit.nn.rnn import LSTMCell

class SimpleLSTM:
    def __init__(self, n_time_step, n_x, n_a, n_y, m):
        self.n_time_step = n_time_step

        # Initialize the activations and predictions
        self.a = np.zeros((n_a, m, n_time_step))
        self.c = np.zeros((n_a, m, n_time_step))
        self.y_pred = np.zeros((n_y, m, n_time_step))

        # Initialize the LSTM cell
        self.lstm_cell = LSTMCell(n_x, n_a, n_y, m)

    def forward_pass(self, xt, a_start=None, c_start=None):
        # Overwrite the activations of the first cell if required
        if a_start is not None:
            self.lstm_cell.a_prev = a_start
            self.lstm_cell.c_prev = c_start

        # Loop over all time steps using the activations of the previous cell and passing the current cell
        # the input for the current time step
        for t in range(self.n_time_step):
            self.a[:, :, t], self.y_pred[:, :, t], self.c[:, :, t] = self.lstm_cell.forward_pass(xt[:, :, t])

        # Return activations and predictions
        return self.a, self.y_pred, self.c


if __name__ == '__main__':
    RANDOM_STATE = 1
    NUM_OBSERVATIONS = 10
    NUM_FEATURES = 3
    NUM_OUTPUT = 2
    NUM_HIDDEN = 5
    NUM_TIMESTEPS = 7

    # Initialize the data
    np.random.seed(RANDOM_STATE)
    x = np.random.randn(NUM_FEATURES, NUM_OBSERVATIONS, NUM_TIMESTEPS)

    # Initialise the SimpleRNN
    lstm = SimpleLSTM(NUM_TIMESTEPS, NUM_FEATURES, NUM_HIDDEN, NUM_OUTPUT, NUM_OBSERVATIONS)

    # Perform one forward pass
    a_next, y_pred, c_next = lstm.forward_pass(x)
    print("a[4][3][6] = ", a_next[4][3][6])
    print("a.shape = ", a_next.shape)
    print("y_pred[1][4][3] =", y_pred[1][4][3])
    print("y_pred.shape = ", y_pred.shape)
    print("c[1][2][1] =", c_next[1][2][1])

    # And another
    # a_next, y_pred = lstm.forward_pass(x, a_next[:,:,-1], c_next[:,:,-1])
    # a_next, y_pred, c_next = lstm.forward_pass(x)
    # print("a[4][1] = ", a_next[4][1])
    # print("a.shape = ", a_next.shape)
    # print("y_pred[1][3] =", y_pred[1][3])
    # print("y_pred.shape = ", y_pred.shape)
