import copy
import json
import jsonlines
import logging
import nltk
import os
import random
import re
import string
import time


PAPERS = [
"art_papers.jsonl",
"econ_papers.jsonl",
"geo_papers.jsonl",
"hist_papers.jsonl",
"phil_papers.jsonl",
]

PRESELECTED = 'data/S2ORC/preselected/'
SAVE_DIR = "data/S2ORC/labeled/" # 5000 papers per field


def randomize(papers_path: str, labeled_save_path: str,
              to_be_mixed_save_path: str):
    logging.info(f"processing: {papers_path}")
    start_time = time.process_time()
    ids = []
    with jsonlines.open(papers_path) as source:
        for line in source.iter():
            ids.append(line['paper_id'])

    random.seed(13)
    random.shuffle(ids)

    # 1000 papers to be mixed with papers from other fields
    mix = ids[-1000:]
    # 2000 of the remaining papers with their own abstracts
    same = ids[:2000]
    # 2000 of the remaining papers to be mixed within the field
    field = ids[2000:4000]

    samesies = []
    field_abstracts = []
    field_papers = []

    with jsonlines.open(papers_path) as source:
        with open(to_be_mixed_save_path, "w") as out:
            for line in source.iter():
                id = line['paper_id']
                abstract = line['abstract']
                text = line['text']
                if id in same:
                    samesies.append({
                        'abstract': abstract,
                        'text': text,
                        'label': 0.95
                    })
                elif id in field:
                    field_abstracts.append(abstract)
                    field_papers.append(text)
                elif id in mix:
                    out.write(json.dumps(line) + "\n")


    fieldies = []
    fieldies.append({
        'abstract': field_abstracts[0],
        'text': field_papers[-1],
        'label': 0.5,
    })
    for i, abstr in enumerate(field_abstracts[1:]):
        paper = field_papers[i]
        fieldies.append({
            'abstract': abstr,
            'text': paper,
            'label': 0.5, 
        })
    with open(labeled_save_path, 'w') as sink:
        for s in samesies:
            sink.write(json.dumps(s) + "\n")
        for f in fieldies:
            sink.write(json.dumps(f) + "\n")
    logging.info(f"within-field labeled data saved in {time.process_time() - start_time}")


def main():

    # randomize and assign labels to papers within the same field
    for i, paper in enumerate(PAPERS):
        randomize(PRESELECTED + paper, SAVE_DIR + f"labeled_{i}.jsonl", SAVE_DIR + f"misc_{i}.jsonl")

    # mix papers from different fields
    logging.info("processing papers to be mixed among fields")
    start_time = time.process_time()
    misc = [file for file in os.listdir(SAVE_DIR) if 'misc' in file]

    full_texts = []
    abstracts = []
    boundaries = [0]
    count = 0
    for i, m in enumerate(misc):
        path = SAVE_DIR + m
        abstr = []
        text = []
        with jsonlines.open(path) as source:
            for line in source.iter():
                count += 1
                abstr.append(line['abstract'])
                text.append(line['text'])
            full_texts.append(text)
            abstracts.append(abstr)
            boundaries.append(count)
    assert len(full_texts) == len(abstracts)
    assert len(full_texts[0]) == len(abstracts[0])
    logging.info(f"all {count} misc papers extracted in {time.process_time() - start_time}")
    start_time = time.process_time()

    random.seed(13)
    shuffled = copy.deepcopy(abstracts)
    random.shuffle(shuffled)
    assert abstracts != shuffled
    assert abstracts[1][0] != shuffled[1][0]

    save_path = SAVE_DIR + "labeled_mixed.jsonl"
    with open(save_path, "w") as sink:
        for i in range(len(full_texts)):
            for idx, text in enumerate(full_texts[i]):
                abstract = shuffled[i][idx]
                dictionary = {
                    'abstract': abstract,
                    'text': text,
                    'label': 0.05
                }
                sink.write(json.dumps(dictionary) + "\n")
    logging.info(f"misc papers shuffled and saved in {time.process_time() - start_time}")


if __name__=='__main__':
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    main()

