import os
import json
from torchnlp.download import download_file_maybe_extract


def squad_dataset(directory='data/',
                  train=False,
                  dev=False,
                  train_filename='train-v2.0.json',
                  dev_filename='dev-v2.0.json',
                  check_files_train=['train-v2.0.json'],
                  check_files_dev=['dev-v2.0.json'],
                  url_train='https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json',
                  url_dev='https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json'):
    """

    Load the Stanford Question Answering Dataset (SQuAD) dataset.
    Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions
    posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment
    of text, or span, from the corresponding reading passage, or the question might be unanswerable.
    SQuAD2.0 combines the 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions written
    adversarially by crowdworkers to look similar to answerable ones. To do well on SQuAD2.0, systems must
    not only answer questions when possible, but also determine when no answer is supported by the paragraph
    and abstain from answering.

    **Reference:** https://rajpurkar.github.io/SQuAD-explorer/
    **Citation:**
    Rajpurkar, P., Jia, R. and Liang, P., 2018. Know what you don't know: Unanswerable questions for SQuAD.
    arXiv preprint arXiv:1806.03822.

    Args:
    directory (str, optional): Directory to cache the dataset.
    train (bool, optional): If to load the training split of the dataset.
    dev (bool, optional): If to load the development split of the dataset.
    train_filename (str, optional): The filename of the training split.
    dev_filename (str, optional): The filename of the development split.
    check_files_train (list, optional):All train filenames
    check_files_dev (list, optional):All development filenames
    url_train (str, optional): URL of the train dataset `.json` file.
    url_dev (str, optional): URL of the dev dataset `.json` file.
    """

    download_file_maybe_extract(url=url_dev, directory=directory, check_files=check_files_dev)
    download_file_maybe_extract(url=url_train, directory=directory, check_files=check_files_train)

    ret = []
    splits = [(train, train_filename), (dev, dev_filename)]
    splits = [f for (requested, f) in splits if requested]
    for filename in splits:
        full_path = os.path.join(directory, filename)
        examples = []
        with open(full_path, 'r') as temp:
            dataset = json.load(temp)

        for article in dataset['data']:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    question = qa['question']
                    answer = [a['text'] for a in qa['answers']]
                    examples.append({
                        'question': question,
                        'answer': answer
                    })
        ret.append(examples)

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)


if __name__ == '__main__':
    train, dev = squad_dataset(train=True, dev=True)
    print(train[5])
    print(len(train))
    print(len(dev))
