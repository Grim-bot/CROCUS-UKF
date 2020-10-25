# CROCUS-UKF
Code to apply the Unscented Kalman Filter (UKF) and the Extended Kalman Filter (EKF) to a measurement from the experimental zero-power reactor CROCUS for pseudo-live reactivity estimation and simultaneous reactor parameter estimation.

## Installation

Install this project by installing all dependencies and subsequently cloning the repository.

#### 1. Non-python dependencies:
You need a working latex installation. You might also need some or all of the following packages: `dvipng texlive-latex-extra texlive-fonts-recommended`. Linux example (Windows with a working latex installation is fine, too):

```sudo apt-get install texlive dvipng texlive-latex-extra texlive-fonts-recommended```

#### 2. Python dependencies:
The code runs under `Python 3.8.5`, as well as `Python 3.7.3.final.0`. Tests are run automatically with `Python 3.8.5` after every `push` or `pull request` to `master`.
The dependencies are listed in [`requirements.txt`](requirements.txt) (respectively [`requirements_3-7-3.txt`](requirements_3-7-3.txt) for `Python 3.7.3.final.0`).

Note: This project depends on [`pykalman`](https://pykalman.github.io), which is not listed in the channels that Anaconda uses by default. If you use Anaconda, use `conda config --append channels conda-forge`.

Note: In order to run tests, such as the ones in `test_RKKF.py`, you also need `pytest`: ```pip install -U pytest``` or ```conda install pytest```.

**With Pip:**

Python 3.8.5:

```pip install -r requirements.txt```

Python 3.7.3:

```pip install -r requirements_3-7-3.txt```

**With conda:**

Python 3.8.5:

```
conda config --append channels conda-forge
conda install --file requirements.txt
```

Python 3.7.3:

```
conda config --append channels conda-forge
conda install --file requirements_3-7-3.txt
```

#### 3. Cloning the repository

```git clone https://github.com/Grim-bot/CROCUS-UKF.git``` and you're good to go!

## Example

#### Running a simulation and generating figures

Once you have installed dependencies and cloned the repository, simply run the `main` function of [`Reactor_Kinetics_Kalman_Filter.py`](Reactor_Kinetics_Kalman_Filter.py). If you run it in an IPython shell (e.g., using an IDE), the figures will appear in the console. Regardless, all figures are saved in pdf format.

```python Reactor_Kinetics_Kalman_Filiter.py```

#### Running tests with `pytest`

```pytest -x```

## Contributing

Have a look at the [Issues](https://github.com/Grim-bot/CROCUS-UKF/issues)! You can also contribute by suggesting a code documentation framework.

## Repository structure
