## About ##

CHARMM GPU-only molecular dynamics package.
N.B. : This is version 0.0.1 of the code. We are adding some bug-fixes right now (like local file radings, constant save intervals etc). Please wait for this message to go away before using apoCHARMM for your prodcution runs. 

## License ##

chcuda is distributed under the
[BSD-3-clause](https://opensource.org/licenses/BSD-3-Clause) open source
license, as described in the LICENSE file in the top level of the repository.
Some external dependencies are used that are licensed under different terms, as
enumerated below.

## Dependencies ##
* [Catch2](https://github.com/catchorg/Catch2) for unit testing
  [(BSL license)](https://opensource.org/licenses/BSL-1.0).

## Authors ##

Samarjeet Prasad (NIH)

Antti-Pekka Hynninen
Bernard R. Brooks (NIH)
# apocharmm


## Installation 

Requirements:
* gcc [10.1]
* CUDA [11.1.1]
* conda's netcdf
* cmake 

<!-- gcc/CUDA pairs working : 10.1//11.1.1 -->

Suggested installation method using [conda](https://conda.io):

1. Clone this repository via 
```
git clone git@github.com:samarjeet/apocharmm --recursive
```
(if you already cloned this repo, but forgot the `--recursive` flag, simply run `git submodule update --init --force --remote` from within the apocharmm directory).

2. Create & activate the right conda environment : 
```
conda env create -f environment.yml
conda activate apoenv
```

3. Finish setting up your compilation environment (make sure you have gcc [10.1] and cuda [11.1.1] available). We suggest using, for faster compilation, `export MAKEFLAGS=-j`.

4. Run the installation via : 

```
pip install -e . 
```

# Generating the Python API documentation

Requires [sphinx](https://www.sphinx-doc.org/en/master/usage/quickstart.html) and its "rtd" theme (both available through conda, and installed by default if you created a conda environment using the `environment.yml` file for the full installation).

Generation of the API documentation requires a GPU machine <!-- because sphinx
is importing the module itself -->. It also requires the apocharmm package to
be installed, i.e. the installation above to have been run.
From the `docs` folder, run 
```
make html
```

Once run, open `docs/build/html/index.html`.

NB: (for dev usage) changes to the documentation require a rebuild to be taken into account.

# Generating the core documentation

Requires [Doxygen](https://doxygen.nl) (`conda install -c conda-forge doxygen`).
To generate the core documentation, run `doxygen docs/doxygen/config/doxybis` from the base directory, then open `docs/doxygen/build/html/index.html`.
The nice HTML theme is [doxygen-awesome](https://github.com/jothepro/doxygen-awesome-css).


## Testing

Prepare tests:

* Create and move to a new directory (`mkdir debug; cd debug`)
* Run `cmake ..` to setup the compilation within this subdir
* Run `make` (same requirements as for the Installation)

Tests executable should be located within `debug/unittests/`
