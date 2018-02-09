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

        # Initialize backprop properties
        self.d_x = np.zeros((n_x, m, n_time_step))
        self.d_W_ax = np.zeros((n_a, n_x))
        self.d_W_aa = np.zeros((n_a, n_a))
        self.d_b_a = np.zeros((n_a, 1))
        self.d_a0 = np.zeros((n_a, m))
        self.d_a_prevt = np.zeros((n_a, m))

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

    def backprop(self, da):

        for t in reversed(range(self.n_time_step)):
            print("self.d_a_prevt")
            print(self.d_a_prevt)
            print()

            grads = self.rnn_cell.backprop(da[:,:,t] + self.d_a_prevt)

            self.d_x[:, :, t] = grads["d_x"]
            self.d_W_ax += grads["d_W_ax"]
            self.d_W_aa += grads["d_W_aa"]
            self.d_b_a += grads["d_b_a"]
            self.d_a_prevt = grads["d_a_prev"]

            print("grads[\"d_x\"]")
            print(grads["d_x"])
            print()

            print("d_W_ax")
            print(grads["d_W_ax"])
            print()

            print("d_W_aa")
            print(grads["d_W_aa"])
            print()

            print("d_b_a")
            print(grads["d_b_a"])
            print()

            print("d_a_prevt")
            print(grads["d_a_prev"])
            print()


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
    a_next, y_pred = rnn.forward_pass(x)

    # print("a[4][1] = ", a_next[4][1])
    # print("a.shape = ", a_next.shape)
    # print("y_pred[1][3] =", y_pred[1][3])
    # print("y_pred.shape = ", y_pred.shape)

    # And another
    # a_next, y_pred = rnn.forward_pass(x, a_next[:,:,-1])
    # a_next, y_pred = rnn.forward_pass(x)
    # print("a[4][1] = ", a_next[4][1])
    # print("a.shape = ", a_next.shape)
    # print("y_pred[1][3] =", y_pred[1][3])
    # print("y_pred.shape = ", y_pred.shape)

    # Perform a backward pass
    da = np.random.randn(5, 10, 4)
    gradients = rnn.backprop(da)

    # print("\nd_x[1][2] =", gradients["d_x"][1][2])
    # print("d_x.shape =", gradients["d_x"].shape)
    # print("d_a0[2][3] =", gradients["d_a0"][2][3])
    # print("d_a0.shape =", gradients["d_a0"].shape)
    # print("d_W_ax[3][1] =", gradients["d_W_ax"][3][1])
    # print("d_W_ax.shape =", gradients["d_W_ax"].shape)
    # print("d_W_aa[1][2] =", gradients["d_W_aa"][1][2])
    # print("d_W_aa.shape =", gradients["d_W_aa"].shape)
    # print("d_b_a[4] =", gradients["d_b_a"][4])
    # print("d_b_a.shape =", gradients["d_b_a"].shape)