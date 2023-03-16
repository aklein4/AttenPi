

class Config(dict):
    def __getattr__(self, attr):
        return self[attr]


DefaultLatentPolicy = Config(
    temp = 0.25,
    dagger_beta = 0.25,

    state_size = 4,
    action_dim = 1,
    action_size = 2,

    num_skills = 2,
    skill_len = 4,

    h_dim = 16,
    action_embed_dim = 4,
    skill_embed_dim = 4,

    num_heads_monitor = 2,
    dim_feedforward_monitor = 32,
    num_layers_monitor = 2,

    num_layers = 2,
    dropout = 0.1
)


CartpolePolicy = DefaultLatentPolicy

WalkerPolicy = Config(
    temp = 0.25,
    
    state_size = 24,
    action_dim = 4,
    action_size = 3,

    num_skills = 4,
    skill_len = 8,

    h_dim = 128,
    action_embed_dim = 32,

    num_heads = 16,
    dim_feedforward = 256,
    num_layers = 4,

    num_heads_monitor = 16,
    dim_feedforward_monitor = 256,
    num_layers_monitor = 4,

    num_layers_chooser = 4,
    dropout_chooser = 0.1,

    num_layers_opter = 4,
    dropout_opter = 0.1
)
