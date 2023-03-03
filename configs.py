

class Config(dict):
    def __getattr__(self, attr):
        return self[attr]


DefaultTrajectory2Policy = Config(
    state_size = 4,
    action_size = 2,

    seq_len = 16,
    h_dim = 64,

    num_encoding_heads = 8,
    num_encoding_layers = 4,
    dim_encoding_feedforward = 128,

    num_decoding_heads = 8,
    num_decoding_layers = 4,
    dim_decoding_feedforward = 128,

    temporal_encoding = False,
    temporal_decoding = False,
    remember_past = False
)


AcrobatTrajectory2Policy = Config(
    state_size = 7,
    action_size = 3,

    seq_len = 64,
    h_dim = 16,

    num_encoding_heads = 8,
    num_encoding_layers = 4,
    dim_encoding_feedforward = 128,

    num_decoding_heads = 8,
    num_decoding_layers = 4,
    dim_decoding_feedforward = 128,

    temporal_encoding = False,
    temporal_decoding = False,
    remember_past = False
)