# NOTE: First, update the version number in `torchnlp.__init__.__version__``
# REFERENCE: https://packaging.python.org/tutorials/distributing-packages/

# Delete the last wheel
rm -r dist/

# Create a source distribution
python3.6 setup.py sdist

# Create a wheel for the project
python3.6 setup.py bdist_wheel

# Upload your distributions to PyPI using twine.
python3.6 -m twine upload dist/*
