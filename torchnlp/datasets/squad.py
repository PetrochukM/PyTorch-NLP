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
                  url_dev='https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json',
                  fine_grained=False):
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
    print(len(train))
    print(len(dev))
