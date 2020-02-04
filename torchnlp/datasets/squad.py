import os
import json
from torchnlp.download import download_files_maybe_extract

def squad_dataset(directory='data/',
				  train=False,
				  dev=False,
				  train_filename='train-v2.0.json',
				  dev_filename='dev-v2.0.json',
				  check_files=['train-v2.0.json',
				  				'dev-v2.0.json'],
				  urls=[
				  		'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json',
				  		'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json'
				  		],
				  fine_grained=False):
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
        test (bool, optional): If to load the test split of the dataset.
        train_filename (str, optional): The filename of the training split.
        dev_filename (str, optional): The filename of the development split.
        test_filename (str, optional): The filename of the test split.
        extracted_name (str, optional): Name of the extracted dataset directory.
        check_files (str, optional): Check if these files exist, then this download was successful.
        url (str, optional): URL of the dataset `.json` file.


    Returns:
        :class:`tuple` of :class:`iterable` or :class:`iterable`:
        Returns between one and all dataset splits (train, dev and test) depending on if their
        respective boolean argument is ``True``.

    Example:
        >>> from torchnlp.datasets import squad_dataset  # doctest: +SKIP
        >>> train = snli_dataset(train=True)  # doctest: +SKIP
        >>> train[0:2]  # doctest: +SKIP
        [{
        'question': 'In what country is Normandy located?', 
        'answer': ['France', 'France', 'France', 'France']
        }, {
        'question': 'When were the Normans in Normandy?', 
        'answer': ['10th and 11th centuries', 'in the 10th and 11th centuries', 
        '10th and 11th centuries', '10th and 11th centuries']
        }]

	"""

	download_files_maybe_extract(urls=urls, directory=directory, check_files=check_files)

	ret = []
	splits = [(train, train_filename), (dev, dev_filename)]
	splits = [f for (requested, f) in splits if requested]
	for filename in splits:
		full_path = os.path.join(directory, extracted_name, filename)
		examples = []
		dataset = json.load(f)

		for article in dataset['data']:
			for paragraph in article['paragraphs']:
				for qa in paragraph['qas']:
					question = qa['question']
					answer = [a['text'] for a in qa['answers']]
					examples.append({
					'question' : question,
					'answer' : answer
					})
		ret.append(examples)

	if(len(ret)==1):
		return ret[0]
	else:
		return tuple(ret)