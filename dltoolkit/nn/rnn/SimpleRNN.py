"""Simple RNN network using RNNCell instances"""
from dltoolkit.utils.utils_rnn import *
from dltoolkit.nn.rnn import RNNCell, Activations


class SimpleRNN:
    def __init__(self, n_time_step, n_x, n_a, n_y, m):
        self.n_time_step = n_time_step

        # Initialize the (initial) activations and predictions
        self.a0 = np.random.randn(n_a, m)
        self.a = np.zeros((n_a, m, n_time_step))
        self.y_pred = np.zeros((n_y, m, n_time_step))

        # Initialize the RNN cells
        self.rnn_cells = []
        for t in range(n_time_step):
            if t == 0:
                rnn_cell = RNNCell(n_x, n_a, n_y, True)
                weights = Activations(rnn_cell.W_aa, rnn_cell.W_ax, rnn_cell.W_ay, rnn_cell.bias_a, rnn_cell.bias_y)
            else:
                # Use the weights of the first cell for all other cells
                rnn_cell = RNNCell()
                rnn_cell.set_activations(weights)

            rnn_cell.a_prev = self.a0
            self.rnn_cells.append(rnn_cell)

        # Initialize backprop properties
        self.d_x = np.zeros((n_x, m, n_time_step))
        self.d_W_ax = np.zeros((n_a, n_x))
        self.d_W_aa = np.zeros((n_a, n_a))
        self.d_b_a = np.zeros((n_a, 1))
        self.d_a0 = np.zeros((n_a, m))
        self.d_a_prevt = np.zeros((n_a, m))

    def forward_pass(self, xt):
        """Perform a forward pass looping over all time steps using the activations of the previous cell and
        passing the current cell the input for the current time step
        """
        a_next = self.a0
        for t in range(self.n_time_step):
            a_next, self.y_pred[:, :, t], a_prev = self.rnn_cells[t].forward_pass(xt[:, :, t], a_next)
            self.a[:, :, t] = a_next

        # Return activations and predictions
        return self.a, self.y_pred

    def backprop(self, da):
        """Perform backprop looping over all time steps"""
        for t in reversed(range(self.n_time_step)):
            grads = self.rnn_cells[t].backprop(da[:,:,t] + self.d_a_prevt)

            self.d_x[:, :, t] = grads["d_x"]
            self.d_W_ax += grads["d_W_ax"]
            self.d_W_aa += grads["d_W_aa"]
            self.d_b_a += grads["d_b_a"]
            self.d_a_prevt = grads["d_a_prev"]

        self.d_a0 = grads["d_a_prev"]
        return {"d_x": self.d_x, "d_a0": self.d_a0, "d_W_ax": self.d_W_ax, "d_W_aa": self.d_W_aa,"d_b_a": self.d_b_a}


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
    res_a_next, res_y_pred = rnn.forward_pass(x)

    print("a[4][1] = ", res_a_next[4][1])
    print("a.shape = ", res_a_next.shape)
    print("y_pred[1][3] =", res_y_pred[1][3])
    print("y_pred.shape = ", res_y_pred.shape)

    # Perform backprop
    da = np.random.randn(NUM_HIDDEN, NUM_OBSERVATIONS, NUM_TIMESTEPS)
    gradients = rnn.backprop(da)
    print("\nd_x[1][2] =", gradients["d_x"][1][2])
    print("d_x.shape =", gradients["d_x"].shape)
    print("d_a0[2][3] =", gradients["d_a0"][2][3])
    print("d_a0.shape =", gradients["d_a0"].shape)
    print("d_W_ax[3][1] =", gradients["d_W_ax"][3][1])
    print("d_W_ax.shape =", gradients["d_W_ax"].shape)
    print("d_W_aa[1][2] =", gradients["d_W_aa"][1][2])
    print("d_W_aa.shape =", gradients["d_W_aa"].shape)
    print("d_b_a[4] =", gradients["d_b_a"][4])
    print("d_b_a.shape =", gradients["d_b_a"].shape)
