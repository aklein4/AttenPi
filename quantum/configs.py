

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
    diff_pi = False,

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


CheetahPolicy = Config(
    temp = 1,

    state_size = 17,
    action_dim = 6,
    action_size = 5,

    num_pi = 4,
    skill_len = 8,
    diff_pi = True,

    delta_dim = 32,
    delta_layers = 2,
    delta_dropout = 0.1,

    pi_dim = 32,
    pi_layers = 2,
    pi_dropout = 0.1,

    latent_dim = 4,
    encoder_dim = 23,
    encoder_layers = 2,
    encoder_dropout = 0.1,

    batch_keep = 4
)

KangarooPolicy = Config(
    temp = 1,

    state_size = 128,
    action_dim = 1,
    action_size = 18,

    num_pi = 8,
    skill_len = 16,
    diff_pi = True,

    delta_dim = 256,
    delta_layers = 2,
    delta_dropout = 0.1,

    pi_dim = 256,
    pi_layers = 2,
    pi_dropout = 0.1,

    latent_dim = 16,
    encoder_dim = 128,
    encoder_layers = 2,
    encoder_dropout = 0.1,

    batch_keep = 4
)