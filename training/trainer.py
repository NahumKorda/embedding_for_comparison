import os.path
import pickle
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import TripletLoss
from sentence_transformers.evaluation import TripletEvaluator
from config import Config
from utils.paths import (LOCAL_DATA_DIR,
                         TRAINING_DATA_DIR,
                         TRAIN_DATASET,
                         EVAL_DATASET,
                         TEST_DATASET,
                         MODEL_DIR)
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


"""
Trains a text embedding model applying the HuggingFace SentenceTransformerTrainer class.
Make sure that you install sentence-transformers version 3.1.1. Newer versions have a known bug in the triplet 
evaluator, which is required for out datasets.

Configuration is provided in external YAML file.

This class does not provide inference interface.
"""

data_dir = os.path.join(LOCAL_DATA_DIR, TRAINING_DATA_DIR)

train_path = os.path.join(str(data_dir), TRAIN_DATASET)
val_path = os.path.join(str(data_dir), EVAL_DATASET)
test_path = os.path.join(str(data_dir), TEST_DATASET)


class Trainer:

    def __init__(self, config: Config, train_dataset: Dataset, eval_dataset: Dataset, from_scratch: bool = True):
        self.__config = config
        self.__model = self.__get_model(config, from_scratch)
        self.__trainer = self.__get_trainer(config, train_dataset, eval_dataset)

    def __get_model(self, config: Config, from_scratch: bool):
        if from_scratch:
            return SentenceTransformer(config.model)
        else:
            model_dir = os.path.join(self.__config.output_dir, MODEL_DIR)
            return SentenceTransformer(str(model_dir))


    def __get_trainer(self, config: Config, train_dataset: Dataset, eval_dataset: Dataset):

        loss = TripletLoss(self.__model)

        dev_evaluator = TripletEvaluator(
            anchors=eval_dataset["anchor"],
            positives=eval_dataset["positive"],
            negatives=eval_dataset["negative"],
            name="pwiz.ai",
            show_progress_bar=False
        )
        dev_evaluator(self.__model)

        return SentenceTransformerTrainer(
            model=self.__model,
            args=config.get_training_args(),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            evaluator=dev_evaluator,
        )

    def train(self):
        self.__trainer.train()

    def evaluate(self, test_dataset: Dataset):
        test_evaluator = TripletEvaluator(
            anchors=test_dataset["anchor"],
            positives=test_dataset["positive"],
            negatives=test_dataset["negative"],
            name="pwiz.ai",
            show_progress_bar=False
        )
        metrics = test_evaluator(self.__model)
        print(metrics)

    def save(self):
        output_dir = os.path.join(self.__config.output_dir, MODEL_DIR)
        self.__model.save_pretrained(output_dir)


def load_local_data(file_path: str) -> list[dict[str, str | list[str]]]:
    with open(file_path, "rb") as input_file:
        return pickle.load(input_file)

def get_local_data(file_path: str) -> Dataset:
    raw_data = load_local_data(file_path)
    anchors, positives, negatives = list(), list(), list()
    for i in range(len(raw_data)):
        example = raw_data[i]
        anchors.append(example["anchor"])
        positives.append(example["positive"])
        negatives.append(example["negative"])
    data = {
        "anchor": anchors,
        "positive": positives,
        "negative": negatives
    }

    return Dataset.from_dict(data)


def train_model():

  train_dataset = get_local_data(train_path)
  eval_dataset = get_local_data(val_path)
  test_dataset = get_local_data(test_path)

  print(f"Train dataset size: {len(train_dataset)}")
  print(f"Eval dataset size: {len(eval_dataset)}")
  print(f"Test dataset size: {len(test_dataset)}")

  config = Config("config.yaml")

  trainer = Trainer(config, train_dataset, eval_dataset)
  trainer.train()
  trainer.save()

  trainer.evaluate(test_dataset)


if __name__ == '__main__':

  train_model()
