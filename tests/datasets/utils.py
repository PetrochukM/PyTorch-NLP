import urllib.request


# Check the URL requested is valid
def urlretrieve_side_effect(url, **kwargs):
    # TODO: Fix failure case if internet does not work
    assert urllib.request.urlopen(url).getcode() == 200
