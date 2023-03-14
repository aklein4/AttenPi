

class Config(dict):
    def __getattr__(self, attr):
        return self[attr]


DefaultLatentPolicy = Config(
    state_size = 4,
    action_size = 2,
    num_skills = 4,

    h_dim = 64,
    max_seq_len = 64,

    num_heads = 16,
    dim_feedforward = 256,
    num_layers = 4,

    num_heads_monitor = 16,
    dim_feedforward_monitor = 256,
    num_layers_monitor = 4
)
