import time
import datetime
import os.path
import pickle
from random import shuffle
from Levenshtein import distance
from tqdm import tqdm
from utils.paths import (LOCAL_DATA_DIR,
                         TRAINING_DATA_DIR,
                         POSITIVE_TRAIN_DATA,
                         NEGATIVE_SAMPLES_FILE,
                         TRAIN_DATASET,
                         EVAL_DATASET,
                         TEST_DATASET)

"""
Generates three datasets: train, eval and test.

Combines augmented positive samples with negative samples chosen in such a manner that they are of similar sizes 
as the positive samples, but very different from them as measured by the Levenshtein distance.

Negative sample candidates are from two unrelated corpora (English Wikipedia categories for shorter sentences and 
HuggingFace sentence-transformers/parallel-sentences-talks for longer sentences).
"""

data_dir = os.path.join(LOCAL_DATA_DIR, TRAINING_DATA_DIR)

positive_path = os.path.join(str(data_dir), POSITIVE_TRAIN_DATA)
negative_path = os.path.join(str(data_dir), NEGATIVE_SAMPLES_FILE)


class DataGenerator:

    def __init__(self, positive_file_path: str, negative_file_path: str):
        self.__positives = self.__get_positives(positive_file_path)
        self.__negatives = self.__get_negatives(negative_file_path)

    @staticmethod
    def __get_positives(positive_file_path: str) -> list[list[str]]:
        with open(positive_file_path, "rb") as input_file:
            return pickle.load(input_file)

    @staticmethod
    def __get_negatives(negative_file_path: str) -> list[str]:
        with open(negative_file_path, "rb") as input_file:
            return pickle.load(input_file)

    def generate_data(self, data_directory: str):

        # Split positives into future datasets
        # Split must be carried out on the raw data
        # to ensure that val and test data are not visible during training
        train_positives, val_positives, test_positives = self.__split_datasets()

        # Generate datasets
        train_dataset = self.__generate_dataset(train_positives)
        val_dataset = self.__generate_dataset(val_positives)
        test_dataset = self.__generate_dataset(test_positives)

        # Save datasets
        file_path = os.path.join(data_directory, TRAIN_DATASET)
        self.__save(train_dataset, file_path)
        file_path = os.path.join(data_directory, EVAL_DATASET)
        self.__save(val_dataset, file_path)
        file_path = os.path.join(data_directory, TEST_DATASET)
        self.__save(test_dataset, file_path)

        return f"Train dataset size: {len(train_dataset)}\nVal dataset size: {len(val_dataset)}\nTest dataset seize: {len(test_dataset)}"

    def __split_datasets(self) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:

        """
        Split raw data by 8:1:1 ratio.
        """
        total = len(self.__positives)
        val_cutoff = int(0.1 * total)
        test_cutoff = val_cutoff + int(0.1 * total)

        shuffle(self.__positives)

        val_positives = self.__positives[:val_cutoff]
        test_positives = self.__positives[val_cutoff: test_cutoff]
        train_positives = self.__positives[test_cutoff:]

        return train_positives, val_positives, test_positives

    def __generate_dataset(self, positives: list[list[str]]):

        retval = list()

        for positive_alternatives in tqdm(positives, total=len(positives)):
            retval.extend(self.__generate_data_points(positive_alternatives))

        return retval

    def __generate_data_points(self, positive_alternatives: list[str]) -> list[dict]:

        retval = list()

        # The first is presumed the original sentence
        positive = positive_alternatives.pop(0)

        negatives = self.__select_negatives(positive, len(positive_alternatives))
        if len(negatives) < len(positive_alternatives):
            negatives = self.__select_random_negative(positive, len(positive_alternatives))

        if len(negatives) < len(positive_alternatives):
            return retval

        for i in range(len(positive_alternatives)):
            retval.append({
                "anchor": positive,
                "positive": positive_alternatives[i],
                "negative": negatives[i]
            })

        return retval

    def __select_negatives(self, positive: str, total_negatives: int) -> list[str]:

        # Calculate the range within which the negative samples must fit.
        lower_bound = len(positive) - int(0.1 * len(positive))
        upper_bound = len(positive) + int(0.1 * len(positive))
        # To accept a negative sample, at least 80% of additions,
        # replacement or deletions must be
        # applied to it to match the positive sample.
        score_threshold = int(0.8 * len(positive))

        # Randomize the order, since we are iterating over it
        # and don't want identical negative samples for similar positive samples.
        shuffle(self.__negatives)

        retval = list()
        for negative in self.__negatives:
            if lower_bound <= len(negative) <= upper_bound:
                score = distance(s1=positive, s2=negative)
                if score > score_threshold:
                    retval.append(negative)
                    if len(retval) == total_negatives:
                        break

        return retval

    def __select_random_negative(self, positive: str, total_negatives: int) -> list[str]:

        """
        This is the "last resort" solution when no suitable negative samples could be matched.
        The loosen criterion is the sample size, but the Levenshtein distance threshold is preserved.
        """

        score_threshold = int(0.5 * len(positive))

        retval = list()
        for negative in self.__negatives:
            score = distance(s1=positive, s2=negative)
            if score > score_threshold:
                retval.append(negative)
                if len(retval) == total_negatives:
                    break

        return retval

    @staticmethod
    def __save(dataset: list[dict[str, str | list[str]]], file_path: str):
        with open(file_path, "wb") as output_file:
            pickle.dump(dataset, output_file)


if __name__ == '__main__':

    start = time.time()

    data_generator = DataGenerator(positive_path, negative_path)

    stats = data_generator.generate_data(str(data_dir))
    print(stats)

    end = time.time()
    diff = end - start
    print("Processing completed in " + str(datetime.timedelta(seconds=diff)))
