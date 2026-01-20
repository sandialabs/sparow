[![Pytest Tests](https://github.com/sandialabs/sparow/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/sandialabs/sparow/actions/workflows/pytest.yml?query=branch%3Amain)
[![Coverage Status](https://github.com/sandialabs/sparow/actions/workflows/coverage.yml/badge.svg?branch=main)](https://github.com/sandialabs/sparow/actions/workflows/coverage.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/sandialabs/sparow/branch/main/graph/badge.svg)](https://codecov.io/gh/sandialabs/sparow)
[![Documentation Status](https://readthedocs.org/projects/sparow/badge/?version=latest)](http://sparow.readthedocs.org/en/latest/)
[![GitHub contributors](https://img.shields.io/github/contributors/sandialabs/sparow.svg)](https://github.com/sandialabs/sparow/graphs/contributors)
[![Merged PRs](https://img.shields.io/github/issues-pr-closed-raw/sandialabs/sparow.svg?label=merged+PRs)](https://github.com/sandialabs/sparow/pulls?q=is:pr+is:merged)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# sparow

## Description

Sparow is a Python library of optimization solvers for stochastic
programming.

## Installation

Sparow requires the installation of or-topas, which can be installed using **pip** as follows:

```bash
git clone git@github.com:or-fusion/or_topas.git
cd or_topas
pip install -e .
cd ..
```

Developers should install Sparow using **pip** as follows:

```bash
git clone git@github.com:or-fusion/sparow.git
cd sparow
pip install -e .
cd ..
```

## Testing

The Pytest software can be used to automatically run all tests within the current directory:

```bash
pytest .
```

Additionally, the following syntax generates a summary that includes code coverage:

```bash
pytest --cov-report term-missing --cov=sparow .
```

## Getting started

TODO

## License

TBD (probably BSD)
