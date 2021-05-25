import argparse
import json
import jsonlines
import logging
import nltk
import os
import re
import string
import time


META_DIR = "data/S2ORC/downloaded/metadata/"
PDF_DIR = "data/S2ORC/downloaded/pdf_parses/"

METADATA = [
'art_papers_meta.txt',
'hist_papers_meta.txt',
'phil_papers_meta.txt',
'geo_papers_meta.txt',
'econ_papers_meta.txt',
]

PAPERS = [
"art_papers.jsonl", # total 6506 papers with len(text) >= 1000 toks and len(abstract) >= 50 toks
"econ_papers.jsonl", # total 144,441
"geo_papers.jsonl", # total 113,755
"hist_papers.jsonl", # total 8474
"phil_papers.jsonl", # total 8316
]
COMPILED = 'data/S2ORC/compiled/'
PRESELECTED = 'data/S2ORC/preselected/'


def _cleanup(text: str) -> str:
    # Removing html tags
    sentence = re.sub(r'<[^>]+>', '', text)

    # Remove parentheticals
    sentence = re.sub("\([^\(]+\)", '', sentence)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z.\'\-]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    # Remove spaces before commas and periods
    sentence = re.sub(r"\s\.", '.', sentence)

    return sentence.strip()


def truncate(text: str) -> str:
    # tokenize
    tokens = nltk.word_tokenize(text)
    # truncate long texts
    if len(tokens) <= 4096:
        return text
    boundary = 0
    for x in tokens[:4096]:
        boundary += len(x)
        boundary += 1
    return text[:boundary]


def compile_metadata():
    # find papers
    metas = [file for file in os.listdir(META_DIR)]
    with open(COMPILED + METADATA[0], 'w') as art_writer:
        with open(COMPILED + METADATA[1], 'w') as hist_writer:
            with open(COMPILED + METADATA[2], 'w') as phil_writer:
                with open(COMPILED + METADATA[3], 'w') as geo_writer:
                    with open(COMPILED + METADATA[4], 'w') as econ_writer:
                        for m in metas:
                            m_path = META_DIR + m
                            with jsonlines.open(m_path) as f:
                                for line in f.iter():
                                    paper_id = line['paper_id']
                                    field = line['mag_field_of_study']
                                    if not field:
                                        continue
                                    if 'Art' in field and 'Geology' not in field and 'Economics' not in field \
                                            and 'Philosophy' not in field and 'History' not in field:
                                        print(paper_id, file=art_writer)
                                    elif 'History' in field and 'Art' not in field and 'Geology' not in field \
                                            and 'Economics' not in field and 'Philosophy' not in field:
                                        print(paper_id, file=hist_writer)
                                    elif 'Philosophy' in field and 'History' not in field and 'Art' not in field \
                                            and 'Geology' not in field and 'Economics' not in field:
                                        print(paper_id, file=phil_writer)
                                    elif 'Geology' in field and 'Economics' not in field and 'Philosophy' not in field \
                                            and 'History' not in field and 'Art' not in field:
                                        print(paper_id, file=geo_writer)
                                    elif 'Economics' in field and 'Philosophy' not in field \
                                            and 'History' not in field and 'Art' not in field and 'Geology' not in field:
                                        print(paper_id, file=econ_writer)
                            print(f"{m} completed")


def compile_papers(paper_ids_filepath: str, save_path: str):
    logging.info(f"processing: {paper_ids_filepath}")
    ids = []
    with open(paper_ids_filepath, 'r') as source:
        for line in source:
            ids.append(line.strip())
    ids = list(set(ids))

    found = 0
    pdfs = [file for file in os.listdir(PDF_DIR)]
    with open(save_path, 'w') as sink:
        for pdf in pdfs:
            pdf_path = PDF_DIR + pdf
            with jsonlines.open(pdf_path) as source:
                for line in source.iter():
                    if not line['abstract'] or not line['body_text']:
                        continue
                    if len(line['abstract']) > 1:
                        continue
                    if line['paper_id'] in ids:
                        abstract = line['abstract'][0]['text']
                        abstract = _cleanup(abstract)
                        if abstract.lower().startswith("abstract"):
                            abstract = abstract[len("abstract") + 1:].strip()
                        text = ''
                        for excerpt in line['body_text']:
                            text += excerpt['text']
                            text += ' '
                        text = _cleanup(text.strip())
                        paper = {
                            'paper_id': line['paper_id'],
                            'abstract': abstract,
                            'text': truncate(text)
                        }
                        sink.write(json.dumps(paper) + "\n")
                        found += 1
            logging.info(f"processed: {pdf}")
    logging.info(f"{paper_ids_filepath}: found {found} papers")
    logging.info("all saved!")


def filter_papers():
    """ Filter to only keep papers with texts >= 1000 tokens and abstracts >= 50 tokens """
    start_time = time.process_time()
    for paper in PAPERS:
        paper_path = COMPILED + paper
        save_path = PRESELECTED + paper
        with jsonlines.open(paper_path) as source:
            count = 0
            with open(save_path, "w") as sink:
                for line in source.iter():
                    text_toks = nltk.word_tokenize(line['text'])
                    if len(text_toks) < 1000:
                        continue
                    abstract_toks = nltk.word_tokenize((line['abstract']))
                    if len(abstract_toks) < 50:
                        continue
                    sink.write(json.dumps(line) + "\n")
                    count += 1
            logging.info(f"{paper_path}: {count} papers selected in {time.process_time() - start_time}")
            start_time = time.process_time()


if __name__=='__main__':
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    compile_metadata()
    for meta, paper in zip(METADATA, PAPERS):
        compile_papers(COMPILED + meta, COMPILED + paper)
    filter_papers
