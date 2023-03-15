

class Config(dict):
    def __getattr__(self, attr):
        return self[attr]


DefaultLatentPolicy = Config(
    temp = 0.25,
    
    state_size = 4,
    action_dim = 1,
    action_size = 2,

    num_skills = 2,
    skill_len = 4,

    h_dim = 16,
    action_embed_dim = 16,

    num_heads = 2,
    dim_feedforward = 32,
    num_layers = 2,

    num_heads_monitor = 2,
    dim_feedforward_monitor = 32,
    num_layers_monitor = 2,

    num_layers_chooser = 2,
    dropout_chooser = 0.1,

    num_layers_opter = 4,
    dropout_opter = 0.1
)


CartpolePolicy = DefaultLatentPolicy

WalkerPolicy = Config(
    temp = 0.25,
    
    state_size = 24,
    action_dim = 4,
    action_size = 3,

    num_skills = 4,
    skill_len = 4,

    h_dim = 64,
    action_embed_dim = 16,

    num_heads = 8,
    dim_feedforward = 128,
    num_layers = 4,

    num_heads_monitor = 8,
    dim_feedforward_monitor = 128,
    num_layers_monitor = 4,

    num_layers_chooser = 4,
    dropout_chooser = 0.1,

    num_layers_opter = 4,
    dropout_opter = 0.1
)
