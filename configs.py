

class Config(dict):
    def __getattr__(self, attr):
        return self[attr]


DefaultLatentPolicy = Config(
    state_size = 6,
    action_size = 2,

    h_dim = 64,
    norm_L = True,

    num_encoding_heads = 8,
    num_encoding_layers = 4,
    dim_encoding_feedforward = 128,

    num_decoding_heads = 8,
    num_decoding_layers = 4,
    dim_decoding_feedforward = 128
)


AcrobatLatentPolicy = Config(
    state_size = 6,
    action_size = 3,

    h_dim = 4,
    norm_L = True,

    num_encoding_heads = 4,
    num_encoding_layers = 3,
    dim_encoding_feedforward = 64,

    num_decoding_heads = 4,
    num_decoding_layers = 3,
    dim_decoding_feedforward = 64,
    norm_l = True
)