import urllib.request


# Check the URL requested is valid
def urlretrieve_side_effect(url, **kwargs):
    assert urllib.request.urlopen(url).getcode() == 200
