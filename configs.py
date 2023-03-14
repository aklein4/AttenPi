

class Config(dict):
    def __getattr__(self, attr):
        return self[attr]


DefaultLatentPolicy = Config(
    state_size = 4,
    action_size = 2,
    num_skills = 4,

    h_dim = 32,
    max_seq_len = 16,

    num_heads = 8,
    dim_feedforward = 128,
    num_layers = 4,

    num_heads_monitor = 8,
    dim_feedforward_monitor = 128,
    num_layers_monitor = 4
)
