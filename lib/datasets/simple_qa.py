import os

import pandas as pd

from lib.datasets.dataset import Dataset

# TODO: Add download simple_qa


def simple_qa_object(directory='data/simple_qa',
                     train=False,
                     dev=False,
                     test=False,
                     train_filename='train.tsv',
                     dev_filename='dev.tsv',
                     test_filename='test.tsv'):
    """
    Sample Data:
        Input: what language is angels vengeance in
        Output: c c c e e c

    # TODO: Update
    """
    ret = []
    datasets = [(train, train_filename), (dev, dev_filename), (test, test_filename)]
    for is_requested, filename in datasets:
        if not is_requested:
            continue
        full_path = os.path.join(directory, filename)
        data = pd.read_table(full_path)
        data = data[data['Object EN Mask'].notnull()]
        # TODO: Investigate data. Look into preprocessing.
        rows = []
        for _, row in data.iterrows():
            rows.append({'source': row['Question EN'].strip(), 'target': row['Object EN Mask']})
        ret.append(Dataset(rows))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)


def simple_qa_predicate(directory='data/SimpleQuestions_v2',
                        train=False,
                        dev=False,
                        test=False,
                        train_filename='annotated_fb_data_train.txt',
                        dev_filename='annotated_fb_data_valid.txt',
                        test_filename='annotated_fb_data_test.txt'):
    """
    Sample Data:
        Input: what is the book e about?
        Output: www.freebase.com/book/written_work/subjects
    """
    ret = []
    datasets = [(train, train_filename), (dev, dev_filename), (test, test_filename)]
    for is_requested, filename in datasets:
        if not is_requested:
            continue
        full_path = os.path.join(directory, filename)
        data = pd.read_table(
            full_path, header=None, names=['subject', 'relation', 'object', 'question'])
        rows = []
        for _, row in data.iterrows():
            rows.append({'text': row['question'].strip(), 'relation': row['relation']})
        ret.append(Dataset(rows))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)


def simple_qa_predicate_preprocessed(directory='data',
                                     train=False,
                                     dev=False,
                                     test=False,
                                     train_filename='preprocessed_question_relation_train.tsv',
                                     dev_filename='preprocessed_question_relation_dev.tsv',
                                     test_filename='preprocessed_question_relation_test.tsv'):
    """
    Sample Data:
        Input: what is the book e about?
        Output: www.freebase.com/book/written_work/subjects
    """
    ret = []
    datasets = [(train, train_filename), (dev, dev_filename), (test, test_filename)]
    for is_requested, filename in datasets:
        if not is_requested:
            continue
        full_path = os.path.join(directory, filename)
        data = pd.read_table(full_path)
        rows = []
        for _, row in data.iterrows():
            rows.append({'text': row['question'].strip(), 'relation': row['relation']})
        ret.append(Dataset(rows))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)


def simple_qa_question_generation(directory='data/simple_qa',
                                  train=False,
                                  dev=False,
                                  test=False,
                                  train_filename='train.tsv',
                                  dev_filename='dev.tsv',
                                  test_filename='test.tsv'):
    """
    Sample Data:
        Input: book/written_work/subjects | E
        Output: what is the book e about
    """
    ret = []
    datasets = [(train, train_filename), (dev, dev_filename), (test, test_filename)]
    for is_requested, filename in datasets:
        if not is_requested:
            continue
        full_path = os.path.join(directory, filename)
        data = pd.read_table(full_path)
        data = data[data['Subject EN'].notnull() & data['Object EN'].notnull()
                    & data['Freebase Property'].notnull()]
        data['Freebase Property'] = data.apply(
            lambda row: row['Freebase Property'].replace('www.freebase.com/', '').strip(), axis=1)
        rows = []
        for _, row in data.iterrows():
            input_ = ' | '.join([row['Subject EN'], row['Object EN'], row['Freebase Property']])
            rows.append({'source': input_, 'target': row['Question EN'].strip()})
        ret.append(Dataset(rows))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
