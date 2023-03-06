

class Config(dict):
    def __getattr__(self, attr):
        return self[attr]


DefaultLatentTrajectory = Config(
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


LunarLatentTrajectory = Config(
    state_size = 8,

    h_dim = 32,
    norm_l = True,
    max_seq_len = 24,

    num_encoding_heads = 32,
    num_encoding_layers = 8,
    dim_encoding_feedforward = 256,

    num_decoding_heads = 32,
    num_decoding_layers = 8,
    dim_decoding_feedforward = 256
)

LunarLatentTrajectoryStable = Config(
    state_size = 8,

    h_dim = 32,
    norm_l = True,
    max_seq_len = 24,

    num_encoding_heads = 32,
    num_encoding_layers = 8,
    dim_encoding_feedforward = 256,

    num_decoding_heads = 32,
    num_decoding_layers = 8,
    dim_decoding_feedforward = 256
)