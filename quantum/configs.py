

class Config(dict):
    def __getattr__(self, attr):
        return self[attr]


DefaultQuantumPolicy = Config(
    temp = 1,

    state_size = 4,
    action_dim = 1,
    action_size = 2,

    num_pi = 3,
    skill_len = 8,

    delta_dim = 16,
    delta_layers = 2,
    delta_dropout = 0.1,

    pi_dim = 16,
    pi_layers = 2,
    pi_dropout = 0.1,

    latent_dim = 4,
    encoder_dim = 16,
    encoder_layers = 2,
    encoder_dropout = 0.1
)


CartpolePolicy = DefaultQuantumPolicy
