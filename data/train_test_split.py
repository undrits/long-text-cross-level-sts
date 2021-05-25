import copy
import json
import jsonlines
import logging
import nltk
import os
import random
import re
import string
from sklearn.model_selection import train_test_split
import time


DIR = "data/S2ORC/labeled/"
SAVE_DIR = 'data/S2ORC/'


"""
INFO: total: 25000 papers
INFO: train: 20250 papers, 440,488,728 tokens
INFO: dev: 2250 papers, 48,814,132 tokens
INFO: test: 2500 papers, 53,942,684 tokens
"""


def main():
    files = [file for file in os.listdir(DIR) if file.startswith("labeled")]

    texts = []
    labels = []
    for file in files:
        filepath = DIR + file
        with jsonlines.open(filepath) as source:
                for line in source.iter():
                    texts.append([line['abstract'], line['text']])
                    labels.append(line['label'])
    assert len(texts) == len(labels)
    logging.info(f"total: {len(texts)} papers")

    x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.1, random_state=13)
    x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.1, random_state=13)

    train_toks = 0
    with open(f"{SAVE_DIR}train.jsonl", "w") as sink:
        for x, y in zip(x_train, y_train):
            d = {
                'abstract': x[0],
                'text': x[1],
                'label': y
            }
            sink.write(json.dumps(d) + "\n")
            train_toks += len(x[0]) + len(x[1])
    logging.info(f"train: {len(x_train)} papers, {train_toks} tokens")

    dev_toks = 0
    with open(f"{SAVE_DIR}dev.jsonl", "w") as sink:
        for x, y in zip(x_dev, y_dev):
            d = {
                'abstract': x[0],
                'text': x[1],
                'label': y
            }
            sink.write(json.dumps(d) + "\n")
            dev_toks += len(x[0]) + len(x[1])
    logging.info(f"dev: {len(x_dev)} papers, {dev_toks} tokens")

    test_toks = 0
    with open(f"{SAVE_DIR}test.jsonl", "w") as sink:
        for x, y in zip(x_test, y_test):
            d = {
                'abstract': x[0],
                'text': x[1],
                'label': y
            }
            sink.write(json.dumps(d) + "\n")
            test_toks += len(x[0]) + len(x[1])
    logging.info(f"test: {len(x_test)} papers, {test_toks} tokens")


if __name__=='__main__':
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    main()
