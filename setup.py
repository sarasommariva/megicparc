from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='megicparc', # Package name
    version='1.0.1',
    description='MEG Informed Cortical Parcellations',
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url='<optional>',
    author='Sara Sommariva',
    author_email='sommariva@dima.unige.it',
    # license='<optional>',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython'
    ],
    # keywords='<optional>',
    packages=find_packages(exclude=['docs', 'tests']),
    # setuptools > 38.6.0 needed for markdown README.md
    setup_requires=['setuptools>=38.6.0'],
)
