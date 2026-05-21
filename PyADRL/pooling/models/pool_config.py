class PoolConfig:
    def __init__(self, map: str, experiment: str, n_pursuers: int, n_evaders: int) -> None:
        self.map = map
        self.experiment = experiment
        self.n_pursuers = n_pursuers
        self.n_evaders = n_evaders
