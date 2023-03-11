

class Config(dict):
    def __getattr__(self, attr):
        return self[attr]


DefaultVariationalTrajectory = Config(
    state_size = 8,
    action_size = 4,

    l_dim = 64,
    max_seq_len = 64,

    num_heads_encoding = 16,
    dim_feedforward_encoding = 256,
    num_layers_encoding = 4,

    num_heads = 16,
    dim_feedforward = 256,
    num_layers = 4
)
