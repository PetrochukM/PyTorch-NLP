import urllib.request
import os
import shutil

import mock

from torchnlp.datasets import imdb_dataset

directory = 'tests/_test_data/'


@mock.patch("urllib.request.urlretrieve")
def test_imdb_dataset_row(mock_urlretrieve):
    # Check the URL requested is valid
    def side_effect(url, **kwargs):
        # TODO: Fix failure case if internet does not work
        assert urllib.request.urlopen(url).getcode() == 200

    mock_urlretrieve.side_effect = side_effect

    # Check a row are parsed correctly
    train, test = imdb_dataset(directory=directory, test=True, train=True)
    assert len(train) > 0
    assert len(test) > 0
    print(test[0])
    assert test[0] == {
        'text':
            "My boyfriend and I went to watch The Guardian.At first I didn't want to watch it, " +
            "but I loved the movie- It was definitely the best movie I have seen in sometime." +
            "They portrayed the USCG very well, it really showed me what they do and I think " +
            "they should really be appreciated more.Not only did it teach but it was a really " +
            "good movie. The movie shows what the really do and how hard the job is.I think " +
            "being a USCG would be challenging and very scary. It was a great movie all around. " +
            "I would suggest this movie for anyone to see.The ending broke my heart but I know " +
            "why he did it. The storyline was great I give it 2 thumbs up. I cried it was very " +
            "emotional, I would give it a 20 if I could!",
        'sentiment':
            'pos'
    }

    # Clean up
    shutil.rmtree(os.path.join(directory, 'aclImdb'))
