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
