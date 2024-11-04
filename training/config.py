from dataclasses import dataclass
import yaml
from sentence_transformers import SentenceTransformerTrainingArguments


@dataclass
class Config:
    model: str
    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    warmup_ratio: float
    fp16: bool
    bf16: bool
    eval_strategy: str
    save_strategy: str
    save_total_limit: int

    def __init__(self, file_path: str):
        with open(file_path, "r") as input_file:
            config = yaml.safe_load(input_file)
        self.model = config["model"]
        self.output_dir = config["output_dir"]
        self.num_train_epochs = config["num_train_epochs"]
        self.per_device_train_batch_size = config["per_device_train_batch_size"]
        self.per_device_eval_batch_size = config["per_device_eval_batch_size"]
        self.learning_rate = config["learning_rate"]
        self.warmup_ratio = config["warmup_ratio"]
        self.fp16 = config["fp16"]
        self.bf16 = config["bf16"]
        self.eval_strategy = config["eval_strategy"]
        self.save_strategy = config["save_strategy"]
        self.save_total_limit = config["save_total_limit"]

    def get_training_args(self):
        return SentenceTransformerTrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            learning_rate=self.learning_rate,
            warmup_ratio=self.warmup_ratio,
            fp16=self.fp16,
            bf16=self.bf16,
            eval_strategy=self.eval_strategy,
            save_strategy=self.save_strategy,
            save_total_limit=self.save_total_limit,
        )
