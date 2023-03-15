

class Config(dict):
    def __getattr__(self, attr):
        return self[attr]


DefaultLatentPolicy = Config(
    state_size = 4,
    action_size = 2,
    num_skills = 4,
    action_temp = 0.25,

    h_dim = 16,
    max_seq_len = 16,

    num_heads = 2,
    dim_feedforward = 32,
    num_layers = 2,

    num_heads_monitor = 2,
    dim_feedforward_monitor = 32,
    num_layers_monitor = 2
)
