import os
import xml.etree.ElementTree as ElementTree
import io
import glob

from torchnlp.download import download_file_maybe_extract
from torchnlp.datasets.dataset import Dataset


def iwslt_dataset(
        directory='data/iwslt/',
        train=False,
        dev=False,
        test=False,
        language_extensions=['en', 'de'],
        train_filename='{source}-{target}/train.{source}-{target}.{lang}',
        dev_filename='{source}-{target}/IWSLT16.TED.tst2013.{source}-{target}.{lang}',
        test_filename='{source}-{target}/IWSLT16.TED.tst2014.{source}-{target}.{lang}',
        check_files=['{source}-{target}/train.tags.{source}-{target}.{source}'],
        url='https://wit3.fbk.eu/archive/2016-01/texts/{source}/{target}/{source}-{target}.tgz'):
    """
    Load the International Workshop on Spoken Language Translation (IWSLT) 2017 translation dataset.

    In-domain training, development and evaluation sets were supplied through the website of the
    WIT3 project, while out-of-domain training data were linked in the workshop’s website. With
    respect to edition 2016 of the evaluation campaign, some of the talks added to the TED
    repository during the last year have been used to define the evaluation sets (tst2017), while
    the remaining new talks have been included in the training sets.

    The English data that participants were asked to recognize and translate consists in part of
    TED talks as in the years before, and in part of real-life lectures and talks that have been
    mainly recorded in lecture halls at KIT and Carnegie Mellon University. TED talks are
    challenging due to their variety in topics, but are very benign as they are very thoroughly
    rehearsed and planned, leading to easy to recognize and translate language.

    References:
      * http://workshop2017.iwslt.org/downloads/iwslt2017_proceeding_v2.pdf
      * http://workshop2017.iwslt.org/

    **Citation:**
    M. Cettolo, C. Girardi, and M. Federico. 2012. WIT3: Web Inventory of Transcribed and Translated
    Talks. In Proc. of EAMT, pp. 261-268, Trento, Italy.

    Args:
        directory (str, optional): Directory to cache the dataset.
        train (bool, optional): If to load the training split of the dataset.
        dev (bool, optional): If to load the dev split of the dataset.
        test (bool, optional): If to load the test split of the dataset.
        language_extensions (:class:`list` of :class:`str`): Two language extensions
            ['en'|'de'|'it'|'ni'|'ro'] to load.
        train_filename (str, optional): The filename of the training split.
        dev_filename (str, optional): The filename of the dev split.
        test_filename (str, optional): The filename of the test split.
        check_files (str, optional): Check if these files exist, then this download was successful.
        url (str, optional): URL of the dataset file.

    Returns:
        :class:`tuple` of :class:`torchnlp.datasets.Dataset`: Tuple with the training tokens, dev
        tokens and test tokens in order if their respective boolean argument is true.

    Example:
        >>> from torchnlp.datasets import iwslt_dataset
        >>> train = iwslt_dataset(train=True)
        >>> train[:2]
        [{
          'en': "David Gallo: This is Bill Lange. I'm Dave Gallo.",
          'de': 'David Gallo: Das ist Bill Lange. Ich bin Dave Gallo.'
        }, {
          'en': "And we're going to tell you some stories from the sea here in video.",
          'de': 'Wir werden Ihnen einige Geschichten über das Meer in Videoform erzählen.'
        }]
    """
    if len(language_extensions) != 2:
        raise ValueError("`language_extensions` must be two language extensions "
                         "['en'|'de'|'it'|'ni'|'ro'] to load.")

    # Format Filenames
    source, target = tuple(language_extensions)
    check_files = [s.format(source=source, target=target) for s in check_files]
    url = url.format(source=source, target=target)

    download_file_maybe_extract(url=url, directory=directory, check_files=check_files)

    iwslt_clean(os.path.join(directory, '{source}-{target}'.format(source=source, target=target)))

    ret = []
    splits = [(train, train_filename), (dev, dev_filename), (test, test_filename)]
    splits = [f for (requested, f) in splits if requested]
    for filename in splits:
        examples = []
        for extension in language_extensions:
            path = os.path.join(directory,
                                filename.format(lang=extension, source=source, target=target))
            with open(path, 'r', encoding='utf-8') as f:
                language_specific_examples = [l.strip() for l in f]

            if len(examples) == 0:
                examples = [{} for _ in range(len(language_specific_examples))]
            for i, example in enumerate(language_specific_examples):
                examples[i][extension] = example

        ret.append(Dataset(examples))

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)


def iwslt_clean(directory):
    # Thanks to torchtext for this snippet:
    # https://github.com/pytorch/text/blob/ea64e1d28c794ed6ffc0a5c66651c33e2f57f01f/torchtext/datasets/translation.py#L152
    for xml_filename in glob.iglob(os.path.join(directory, '*.xml')):
        txt_filename = os.path.splitext(xml_filename)[0]
        if os.path.isfile(txt_filename):
            continue

        with io.open(txt_filename, mode='w', encoding='utf-8') as f:
            root = ElementTree.parse(xml_filename).getroot()[0]
            for doc in root.findall('doc'):
                for element in doc.findall('seg'):
                    f.write(element.text.strip() + '\n')

    xml_tags = [
        '<url', '<keywords', '<talkid', '<description', '<reviewer', '<translator', '<title',
        '<speaker'
    ]
    for original_filename in glob.iglob(os.path.join(directory, 'train.tags*')):
        txt_filename = original_filename.replace('.tags', '')
        if os.path.isfile(txt_filename):
            continue

        with io.open(txt_filename, mode='w', encoding='utf-8') as txt_file, \
                io.open(original_filename, mode='r', encoding='utf-8') as original_file:
            for line in original_file:
                if not any(tag in line for tag in xml_tags):
                    txt_file.write(line.strip() + '\n')
