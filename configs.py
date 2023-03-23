

class Config(dict):
    def __getattr__(self, attr):
        return self[attr]


DefaultQuantumPolicy = Config(
    pred_temp = 0.25,
    grad_temp = 0.25,
    diff_delta = True,

    state_size = 64,
    action_dim = 1,
    action_size = 9,

    num_pi = 4,
    skill_len = 8,

    hidden_dim = 64,
    num_layers = 4,
    dropout = 0.1,
    latent_dim = 16,

    batch_keep = 4
)
