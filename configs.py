

class Config(dict):
    def __getattr__(self, attr):
        return self[attr]


DefaultLatentPolicy = Config(
    temp = 0.25,
    
    state_size = 4,
    action_dim = 1,
    action_size = 2,

    num_skills = 4,
    skill_len = 8,

    h_dim = 16,
    action_embed_dim = 16,

    num_heads = 2,
    dim_feedforward = 32,
    num_layers = 2,

    num_heads_monitor = 2,
    dim_feedforward_monitor = 32,
    num_layers_monitor = 2,

    num_layers_chooser = 2,
    dropout_chooser = 0.1
)


CartpolePolicy = DefaultLatentPolicy