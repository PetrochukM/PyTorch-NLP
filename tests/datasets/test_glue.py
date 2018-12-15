import mock
import shutil

from torchnlp.datasets import glue_dataset
from tests.datasets.utils import urlretrieve_side_effect

GLUE_DIRECTORY = 'tests/_test_data/glue'


@mock.patch("urllib.request.urlretrieve")
def test_glue_dataset_row(mock_urlretrieve):
    mock_urlretrieve.side_effect = urlretrieve_side_effect
    test_CoLA_dataset()
    test_SST_dataset()
    test_MRPC_dataset()
    test_QQP_dataset()
    test_STS_dataset()
    test_MNLI_dataset()
    test_SNLI_dataset()
    test_QNLI_dataset()
    test_RTE_dataset()
    test_WNLI_dataset()
    test_diagnostic_dataset()
    shutil.rmtree(GLUE_DIRECTORY)


def test_CoLA_dataset():
    train, dev, test = glue_dataset(
        directory=GLUE_DIRECTORY, train=True, dev=True, test=True, dataset='CoLA')
    assert len(train) > 0
    assert len(dev) > 0
    assert len(test) > 0
    assert train[0] == {
        'source': 'gj04',
        'acceptability judgment': '1',
        'original acceptability judgment': '',
        'sentence': "Our friends won't buy this analysis, let alone the next one we propose."
    }


def test_SST_dataset():
    train, dev, test = glue_dataset(
        directory=GLUE_DIRECTORY, train=True, dev=True, test=True, dataset='SST')
    assert len(train) > 0
    assert len(dev) > 0
    assert len(test) > 0
    assert train[0] == {'sentence': 'hide new secretions from the parental units ', 'label': '0'}


def test_MRPC_dataset():
    train, dev, test = glue_dataset(
        directory=GLUE_DIRECTORY, train=True, dev=True, test=True, dataset='MRPC')
    assert len(train) > 0
    assert len(dev) > 0
    assert len(test) > 0
    print(train[0])
    assert train[0] == {
        'Quality': '1',
        '#1 ID': '702876',
        '#2 ID': '702977',
        '#1 String': 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
        '#2 String': 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .'
    }


def test_QQP_dataset():
    train, dev, test = glue_dataset(
        directory=GLUE_DIRECTORY, train=True, dev=True, test=True, dataset='QQP')
    assert len(train) > 0
    assert len(dev) > 0
    assert len(test) > 0
    assert train[0] == {
        'id': '133273',
        'qid1': '213221',
        'qid2': '213222',
        'question1': 'How is the life of a math student? Could you describe your own experiences?',
        'question2': 'Which level of prepration is enough for the exam jlpt5?', 'is_duplicate': '0'
    }


def test_STS_dataset():
    train, dev, test = glue_dataset(
        directory=GLUE_DIRECTORY, train=True, dev=True, test=True, dataset='STS')
    assert len(train) > 0
    assert len(dev) > 0
    assert len(test) > 0
    assert train[0] == {
        'index': '0',
        'genre': 'main-captions',
        'filename': 'MSRvid',
        'year': '2012test',
        'old_index': '0001',
        'source1': 'none',
        'source2': 'none',
        'sentence1': 'A plane is taking off.',
        'sentence2': 'An air plane is taking off.',
        'score': '5.000'
    }


def test_MNLI_dataset():
    train, dev_matched, dev_mismatched, test_matched, test_mismatched = glue_dataset(
        directory=GLUE_DIRECTORY, train=True, dev=True, test=True, dev_mismatched=True, test_mismatched=True, dataset='MNLI')
    assert len(train) > 0
    assert len(dev_matched) > 0
    assert len(dev_mismatched) > 0
    assert len(test_matched) > 0
    assert len(test_mismatched) > 0
    assert train[0] == {
        'index': '0',
        'promptID': '31193',
        'pairID': '31193n',
        'genre': 'government',
        'sentence1_binary_parse': '( ( Conceptually ( cream skimming ) ) ( ( has ( ( ( two ( basic dimensions ) ) - ) ( ( product and ) geography ) ) ) . ) )',
        'sentence2_binary_parse': '( ( ( Product and ) geography ) ( ( are ( what ( make ( cream ( skimming work ) ) ) ) ) . ) )',
        'sentence1_parse': '(ROOT (S (NP (JJ Conceptually) (NN cream) (NN skimming)) (VP (VBZ has) (NP (NP (CD two) (JJ basic) (NNS dimensions)) (: -) (NP (NN product) (CC and) (NN geography)))) (. .)))',
        'sentence2_parse': '(ROOT (S (NP (NN Product) (CC and) (NN geography)) (VP (VBP are) (SBAR (WHNP (WP what)) (S (VP (VBP make) (NP (NP (NN cream)) (VP (VBG skimming) (NP (NN work)))))))) (. .)))',
        'sentence1': 'Conceptually cream skimming has two basic dimensions - product and geography.',
        'sentence2': 'Product and geography are what make cream skimming work. ',
        'label1': 'neutral',
        'gold_label': 'neutral'
    }


def test_SNLI_dataset():
    train, dev, test = glue_dataset(
        directory=GLUE_DIRECTORY, train=True, dev=True, test=True, dataset='SNLI')
    assert len(train) > 0
    assert len(dev) > 0
    assert len(test) > 0
    assert train[0] == {
        'index': '0',
        'captionID': '3416050480.jpg#4',
        'pairID': '3416050480.jpg#4r1n',
        'sentence1_binary_parse': '( ( ( A person ) ( on ( a horse ) ) ) ( ( jumps ( over ( a ( broken ( down airplane ) ) ) ) ) . ) )',
        'sentence2_binary_parse': '( ( A person ) ( ( is ( ( training ( his horse ) ) ( for ( a competition ) ) ) ) . ) )',
        'sentence1_parse': '(ROOT (S (NP (NP (DT A) (NN person)) (PP (IN on) (NP (DT a) (NN horse)))) (VP (VBZ jumps) (PP (IN over) (NP (DT a) (JJ broken) (JJ down) (NN airplane)))) (. .)))',
        'sentence2_parse': '(ROOT (S (NP (DT A) (NN person)) (VP (VBZ is) (VP (VBG training) (NP (PRP$ his) (NN horse)) (PP (IN for) (NP (DT a) (NN competition))))) (. .)))',
        'sentence1': 'A person on a horse jumps over a broken down airplane.',
        'sentence2': 'A person is training his horse for a competition.',
        'label1': 'neutral',
        'gold_label': 'neutral'
    }


def test_QNLI_dataset():
    train, dev, test = glue_dataset(
        directory=GLUE_DIRECTORY, train=True, dev=True, test=True, dataset='QNLI')
    assert len(train) > 0
    assert len(dev) > 0
    assert len(test) > 0
    assert train[0] == {
        'index': '0',
        'question': 'What is the Grotto at Notre Dame?',
        'sentence': 'Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection.',
        'label': 'entailment'
    }


def test_RTE_dataset():
    train, dev, test = glue_dataset(
        directory=GLUE_DIRECTORY, train=True, dev=True, test=True, dataset='RTE')
    assert len(train) > 0
    assert len(dev) > 0
    assert len(test) > 0
    assert train[0] == {
        'index': '0',
        'sentence1': 'No Weapons of Mass Destruction Found in Iraq Yet.',
        'sentence2': 'Weapons of Mass Destruction Found in Iraq.',
        'label': 'not_entailment'
    }


def test_WNLI_dataset():
    train, dev, test = glue_dataset(
        directory=GLUE_DIRECTORY, train=True, dev=True, test=True, dataset='WNLI')
    assert len(train) > 0
    assert len(dev) > 0
    assert len(test) > 0
    assert train[0] == {
        'index': '0',
        'sentence1': 'I stuck a pin through a carrot. When I pulled the pin out, it had a hole.',
        'sentence2': 'The carrot had a hole.',
        'label': '1'
    }


def test_diagnostic_dataset():
    train = glue_dataset(
        directory=GLUE_DIRECTORY, train=True, dataset='diagnostic')
    assert len(train) > 0
    assert train[0] == {
        'index': '0',
        'sentence1': 'The cat sat on the mat.',
        'sentence2': 'The cat did not sit on the mat.'
    }


    
