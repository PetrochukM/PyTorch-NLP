from torchnlp.word_to_vector.pretrained_word_vectors import _PretrainedWordVectors

# List of all 275 supported languages from http://cosyne.h-its.org/bpemb/data/
SUPPORTED_LANGUAGES = [
    'ab', 'ace', 'ady', 'af', 'ak', 'als', 'am', 'an', 'ang', 'ar', 'arc', 'arz', 'as', 'ast',
    'atj', 'av', 'ay', 'az', 'azb', 'ba', 'bar', 'bcl', 'be', 'bg', 'bi', 'bjn', 'bm', 'bn', 'bo',
    'bpy', 'br', 'bs', 'bug', 'bxr', 'ca', 'cdo', 'ce', 'ceb', 'ch', 'chr', 'chy', 'ckb', 'co',
    'cr', 'crh', 'cs', 'csb', 'cu', 'cv', 'cy', 'da', 'de', 'din', 'diq', 'dsb', 'dty', 'dv', 'dz',
    'ee', 'el', 'en', 'eo', 'es', 'et', 'eu', 'ext', 'fa', 'ff', 'fi', 'fj', 'fo', 'fr', 'frp',
    'frr', 'fur', 'fy', 'ga', 'gag', 'gan', 'gd', 'gl', 'glk', 'gn', 'gom', 'got', 'gu', 'gv', 'ha',
    'hak', 'haw', 'he', 'hi', 'hif', 'hr', 'hsb', 'ht', 'hu', 'hy', 'ia', 'id', 'ie', 'ig', 'ik',
    'ilo', 'io', 'is', 'it', 'iu', 'ja', 'jam', 'jbo', 'jv', 'ka', 'kaa', 'kab', 'kbd', 'kbp', 'kg',
    'ki', 'kk', 'kl', 'km', 'kn', 'ko', 'koi', 'krc', 'ks', 'ksh', 'ku', 'kv', 'kw', 'ky', 'la',
    'lad', 'lb', 'lbe', 'lez', 'lg', 'li', 'lij', 'lmo', 'ln', 'lo', 'lrc', 'lt', 'ltg', 'lv',
    'mai', 'mdf', 'mg', 'mh', 'mhr', 'mi', 'min', 'mk', 'ml', 'mn', 'mr', 'mrj', 'ms', 'mt', 'mwl',
    'my', 'myv', 'mzn', 'na', 'nap', 'nds', 'ne', 'new', 'ng', 'nl', 'nn', 'no', 'nov', 'nrm',
    'nso', 'nv', 'ny', 'oc', 'olo', 'om', 'or', 'os', 'pa', 'pag', 'pam', 'pap', 'pcd', 'pdc',
    'pfl', 'pi', 'pih', 'pl', 'pms', 'pnb', 'pnt', 'ps', 'pt', 'qu', 'rm', 'rmy', 'rn', 'ro', 'ru',
    'rue', 'rw', 'sa', 'sah', 'sc', 'scn', 'sco', 'sd', 'se', 'sg', 'sh', 'si', 'sk', 'sl', 'sm',
    'sn', 'so', 'sq', 'sr', 'srn', 'ss', 'st', 'stq', 'su', 'sv', 'sw', 'szl', 'ta', 'tcy', 'te',
    'tet', 'tg', 'th', 'ti', 'tk', 'tl', 'tn', 'to', 'tpi', 'tr', 'ts', 'tt', 'tum', 'tw', 'ty',
    'tyv', 'udm', 'ug', 'uk', 'ur', 'uz', 've', 'vec', 'vep', 'vi', 'vls', 'vo', 'wa', 'war', 'wo',
    'wuu', 'xal', 'xh', 'xmf', 'yi', 'yo', 'za', 'zea', 'zh', 'zu'
]

# All supported vector dimensionalities for which embeddings were trained
SUPPORTED_DIMS = [25, 50, 100, 200, 300]

# All supported number of merge operations for which embeddings were trained
SUPPORTED_MERGE_OPS = [1000, 3000, 5000, 10000, 25000, 50000, 100000, 200000]


class BPEmb(_PretrainedWordVectors):
    """
    Byte-Pair Encoding (BPE) embeddings trained on Wikipedia for 275 languages

    A collection of pre-trained subword unit embeddings in 275 languages, based
    on Byte-Pair Encoding (BPE). In an evaluation using fine-grained entity typing as testbed,
    BPEmb performs competitively, and for some languages better than alternative subword
    approaches, while requiring vastly fewer resources and no tokenization.

    References:
        * https://arxiv.org/abs/1710.02187
        * https://github.com/bheinzerling/bpemb

    Args:
        language (str, optional): Language of the corpus on which the embeddings
            have been trained
        dim (int, optional): Dimensionality of the embeddings
        merge_ops (int, optional): Number of merge operations used by the
            tokenizer

    Example:
        >>> from torchnlp.word_to_vector import BPEmb
        >>> vectors = BPEmb(dim=25)
        >>> subwords = "▁mel ford shire".split()
        >>> vectors[subwords]
        Columns 0 to 9
        -0.5859 -0.1803  0.2623 -0.6052  0.0194 -0.2795  0.2716 -0.2957 -0.0492
        1.0934
         0.3848 -0.2412  1.0599 -0.8588 -1.2596 -0.2534 -0.5704  0.2168 -0.1718
        1.2675
         1.4407 -0.0996  1.2239 -0.5085 -0.7542 -0.9628 -1.7177  0.0618 -0.4025
        1.0405
        ...
        Columns 20 to 24
        -0.0022  0.4820 -0.5156 -0.0564  0.4300
         0.0355 -0.2257  0.1323  0.6053 -0.8878
        -0.0167 -0.3686  0.9666  0.2497 -1.2239
        [torch.FloatTensor of size 3x25]
    """
    url_base = 'http://cosyne.h-its.org/bpemb/data/{language}/'
    file_name = '{language}.wiki.bpe.op{merge_ops}.d{dim}.w2v.txt'
    zip_extension = '.tar.gz'

    def __init__(self, language='en', dim=300, merge_ops=50000, **kwargs):
        # Check if all parameters are valid
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(("Language '%s' not supported. Use one of the "
                              "following options instead:\n%s") % (language, SUPPORTED_LANGUAGES))
        if dim not in SUPPORTED_DIMS:
            raise ValueError(("Embedding dimensionality of '%d' not supported. "
                              "Use one of the following options instead:\n%s") % (dim,
                                                                                  SUPPORTED_DIMS))
        if merge_ops not in SUPPORTED_MERGE_OPS:
            raise ValueError(("Number of '%d' merge operations not supported. "
                              "Use one of the following options instead:\n%s") %
                             (merge_ops, SUPPORTED_MERGE_OPS))

        format_map = {'language': language, 'merge_ops': merge_ops, 'dim': dim}

        # Assemble file name to locally store embeddings under
        name = self.file_name.format_map(format_map)
        # Assemble URL to download the embeddings form
        url = (
            self.url_base.format_map(format_map) + self.file_name.format_map(format_map) +
            self.zip_extension)

        super(BPEmb, self).__init__(name, url=url, **kwargs)
