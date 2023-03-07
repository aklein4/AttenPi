

class Config(dict):
    def __getattr__(self, attr):
        return self[attr]


DefaultPathFinder = Config(
    state_size = 8,

    h_dim = 64,
    max_seq_len = 128,
    
    mid_dim = 128,

    num_heads_encoding = 16,
    dim_feedforward_encoding = 256,
    num_layers_encoding = 4,

    num_heads = 16,
    dim_feedforward = 256,
    num_layers = 4
)
