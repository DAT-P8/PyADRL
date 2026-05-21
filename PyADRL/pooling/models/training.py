from pathlib import Path


class Training:
    def __init__(self, name: str, training_location: Path) -> None:
        self.name = name
        self.final_model_path = training_location / "final_model"
        self.models_path = training_location / "models"
        self.figures_path = training_location / "figures"
        self.metrics_path = training_location / "evaluation_metrics"
        self.eval_pool_path = training_location / "eval_pool"
