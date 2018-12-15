import os
import csv

from urllib.parse import urlparse

from torchnlp.datasets.dataset import Dataset
from torchnlp.download import download_file_maybe_extract

ZIP_FOLDER_NAME = {"CoLA":"CoLA", "SST":"SST-2", "MRPC":"MRPC", "QQP":"QQP", "STS":"STS-B", "MNLI":"MNLI", "SNLI":"SNLI", "QNLI":"QNLI", "RTE":"RTE", "WNLI":"WNLI", "diagnostic":"diagnostic"}

GLUE_DATASETS_PATHS = {"CoLA":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FCoLA.zip?alt=media&token=46d5e637-3411-4188-bc44-5809b5bfb5f4',
        "SST":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8',
        "MRPC":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2Fmrpc_dev_ids.tsv?alt=media&token=ec5c0836-31d5-48f4-b431-7480817f1adc',
        "QQP":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQQP.zip?alt=media&token=700c6acf-160d-4d89-81d1-de4191d02cb5',
        "STS":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSTS-B.zip?alt=media&token=bddb94a7-8706-4e0d-a694-1109e12273b5',
        "MNLI":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FMNLI.zip?alt=media&token=50329ea1-e339-40e2-809c-10c40afff3ce',
        "SNLI":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSNLI.zip?alt=media&token=4afcfbb2-ff0c-4b2d-a09a-dbf07926f4df',
        "QNLI":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQNLI.zip?alt=media&token=c24cad61-f2df-4f04-9ab6-aa576fa829d0',
        "RTE":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FRTE.zip?alt=media&token=5efa7e85-a0bb-4f19-8ea2-9e1840f077fb',
        "WNLI":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FWNLI.zip?alt=media&token=068ad0a0-ded7-4bd7-99a5-5e00222e0faf',
        "diagnostic":'https://storage.googleapis.com/mtl-sentence-representations.appspot.com/tsvsWithoutLabels%2FAX.tsv?GoogleAccessId=firebase-adminsdk-0khhl@mtl-sentence-representations.iam.gserviceaccount.com&Expires=2498860800&Signature=DuQ2CSPt2Yfre0C%2BiISrVYrIFaZH1Lc7hBVZDD4ZyR7fZYOMNOUGpi8QxBmTNOrNPjR3z1cggo7WXFfrgECP6FBJSsURv8Ybrue8Ypt%2FTPxbuJ0Xc2FhDi%2BarnecCBFO77RSbfuz%2Bs95hRrYhTnByqu3U%2FYZPaj3tZt5QdfpH2IUROY8LiBXoXS46LE%2FgOQc%2FKN%2BA9SoscRDYsnxHfG0IjXGwHN%2Bf88q6hOmAxeNPx6moDulUF6XMUAaXCSFU%2BnRO2RDL9CapWxj%2BDl7syNyHhB7987hZ80B%2FwFkQ3MEs8auvt5XW1%2Bd4aCU7ytgM69r8JDCwibfhZxpaa4gd50QXQ%3D%3D'}

MRPC_TRAIN = 'https://s3.amazonaws.com/senteval/senteval_data/msr_paraphrase_train.txt'
MRPC_TEST = 'https://s3.amazonaws.com/senteval/senteval_data/msr_paraphrase_test.txt'

def glue_dataset(directory='data/GLUE/',
                 train=False,
                 dev=False,
                 test=False,
                 dev_mismatched=False,
                 test_mismatched=False,
                 train_filename='train.tsv',
                 dev_filename='dev.tsv',
                 test_filename='test.tsv',
                 dev_filename_mismatched='dev_mismatched.tsv',
                 test_filename_mismatched='test_mismatched.tsv',
                 check_files=['train.tsv'],
                 dataset=None):
    """
    Load the Generalized Language Understanding Evaluation (GLUE) benchmark.

    The GLUE benchmark is a collection of resources for training, evaluating, and analyzing natural language understanding systems.

    References:
        - https://github.com/nyu-mll/GLUE-baselines
        - https://www.nyu.edu/projects/bowman/glue.pdf
        - https://gluebenchmark.com

    **Citation**
    ::
        @unpublished{wang2018glue
             title={{GLUE}: A Multi-Task Benchmark and Analysis Platform for
                     Natural Language Understanding}
             author={Wang, Alex and Singh, Amanpreet and Michael, Julian and Hill,
                     Felix and Levy, Omer and Bowman, Samuel R.}
             note={arXiv preprint 1804.07461}
             year={2018}
         }

    Args:
        directory (str, optional): Directory to cache the dataset.
        train (bool, optional): If to load the training split of the dataset.
        dev (bool, optional): If to load the dev split of the dataset.
        test (bool, optional): If to load the test split of the dataset.
        dev_mismatched (bool, optional): Only used for the MNLI set, if to load the dev mismatched split of the dataset.
        test_mismatched (bool, optional): Only used for the MNLI set, if to load the test mismatched split of the dataset.
        train_filename (str, optional): The filename of the training split.
        dev_filename (str, optional): The filename of the development split.
        test_filename (str, optional): The filename of the test split.
        dev_filename_mismatched (str, optional) : Only used for the MNLI set, the filename of the mismatched development split
        test_filename_mismatched (str, optional) : Only used for the MNLI set, the filename of the mismatched test split
        check_files (str, optional): Check if these files exist, then this download was successful.
        dataset (str, optional): Dataset of GLUE to download. 

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset` or :class:`torchnlp.datasets.Dataset`:
        Returns between one and all dataset splits (train, dev and test) depending on if their
        respective boolean argument is ``True``.
        
        remark: when loading the MNLI dataset, values are returned in the following order if their respective boolean arguments are true :
                (train, dev_matched, dev_mismatched, test_matched, test_mismatched)

    Example:
        >>> from torchnlp.datasets import glue_dataset
        >>> train = glue_dataset(train=True, dataset = 'QNLI')
        >>> train[:2]
        [{
          'index': '0', 
          'question': 'What is the Grotto at Notre Dame?', 
          'sentence': 'Immediately behind the basilica ... ', 
          'label': 'entailment'
        }, {
          'index': '1', 
          'question': 'What is the Grotto at Notre Dame?', 
          'sentence': 'It is a replica of the grotto at ... ', 
          'label': 'not_entailment'
        }]
    """
    if dataset == None :
        print('You must select one of the GLUE dataset. (CoLA, SST, MRPC, QQP, STS, MNLI, SNLI, QNLI, RTE, WNLI or diagnostic)')
        return

    assert dataset in ZIP_FOLDER_NAME, "Dataset %s not found!" % dataset
    folder_path = os.path.join(directory,ZIP_FOLDER_NAME[dataset])
    url = GLUE_DATASETS_PATHS[dataset]
    check_file_list = [os.path.join(ZIP_FOLDER_NAME[dataset],f) for f in check_files]
    if dataset == 'MRPC':
        download_file_maybe_extract(url=url, directory=folder_path, filename = 'dev_ids.tsv')
        download_file_maybe_extract(url=MRPC_TRAIN, directory=folder_path, filename = 'msr_paraphrase_train.txt')
        download_file_maybe_extract(url=MRPC_TEST, directory=folder_path, filename = 'msr_paraphrase_test.txt')
        mrpc_train_file = os.path.join(folder_path,'msr_paraphrase_train.txt')
        mrpc_test_file = os.path.join(folder_path,'msr_paraphrase_test.txt')
        mrpc_dev_file = os.path.join(folder_path,'dev_ids.tsv')
        assert os.path.isfile(mrpc_dev_file), "Dev data not found at %s" % mrpc_dev_file
        assert os.path.isfile(mrpc_train_file), "Train data not found at %s" % mrpc_train_file
        assert os.path.isfile(mrpc_test_file), "Test data not found at %s" % mrpc_test_file
        MRPC_processing(folder_path,mrpc_train_file,mrpc_test_file)
        os.remove(mrpc_train_file)
        os.remove(mrpc_test_file)
        os.remove(mrpc_dev_file)
    elif dataset == 'diagnostic':
        download_file_maybe_extract(url=url, directory=folder_path, filename='diagnostic.tsv', check_files=[])
    else:
        download_file_maybe_extract(url=url, directory=directory, check_files=check_file_list)
        parse = urlparse(url)
        zip_file = os.path.join(directory,os.path.basename(parse.path))
        if os.path.isfile(zip_file):
            os.remove(zip_file)
    ret = []
    if dataset == 'MNLI':
        dev_filename = "dev_matched.tsv"
        test_filename = "test_matched.tsv"
        splits = [(train, train_filename), (dev, dev_filename), (dev_mismatched, dev_filename_mismatched), (test, test_filename), (test_mismatched, test_filename_mismatched)]
        splits = [f for (requested, f) in splits if requested]
    else: 
        splits = [(train, train_filename), (dev, dev_filename), (test, test_filename)]
        splits = [f for (requested, f) in splits if requested]

    if dataset == 'diagnostic':
        examples = []
        with open(os.path.join(folder_path,'diagnostic.tsv'), newline='') as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter='\t',quoting=csv.QUOTE_NONE)
            keys = next(tsvreader)
            nb_arguments = len(keys)
            for line in tsvreader:
                examples.append({keys[i]:line[i] for i in range(nb_arguments)})
        ret.append(Dataset(examples))
    else:
        for filename in splits:
            examples = []
            with open(os.path.join(folder_path,filename), newline='') as tsvfile:
                tsvreader = csv.reader(tsvfile, delimiter='\t',quoting=csv.QUOTE_NONE)
                keys = next(tsvreader)
                nb_arguments = len(keys)
                for line in tsvreader:
                    """
                        problem with QQP and SNLI : len(line) is different from nb_arguments for some lines ... 
                    """
                    if len(line) == nb_arguments:
                        examples.append({keys[i]:line[i] for i in range(nb_arguments)})
                    else:
                        pass
            ret.append(Dataset(examples))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
    
def MRPC_processing(folder_path,mrpc_train_file,mrpc_test_file):
    dev_ids = []
    with open(os.path.join(folder_path, "dev_ids.tsv")) as ids_fh:
        for row in ids_fh:
            dev_ids.append(row.strip().split('\t'))

    with open(mrpc_train_file) as data_fh, \
         open(os.path.join(folder_path, "train.tsv"), 'w') as train_fh, \
         open(os.path.join(folder_path, "dev.tsv"), 'w') as dev_fh:
        header = data_fh.readline()
        train_fh.write(header)
        dev_fh.write(header)
        for row in data_fh:
            label, id1, id2, s1, s2 = row.strip().split('\t')
            if [id1, id2] in dev_ids:
                dev_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))
            else:
                train_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))

    with open(mrpc_test_file) as data_fh, \
            open(os.path.join(folder_path, "test.tsv"), 'w') as test_fh:
        header = data_fh.readline()
        test_fh.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for idx, row in enumerate(data_fh):
            label, id1, id2, s1, s2 = row.strip().split('\t')
            test_fh.write("%d\t%s\t%s\t%s\t%s\n" % (idx, id1, id2, s1, s2))

