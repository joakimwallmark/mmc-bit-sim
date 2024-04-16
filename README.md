# Instructions for replicating simulation study
This repository contains the code to replicate the simulation study in the article "Introducing nonparametric monotone multiple choice item response theory models and bit scales" by Wallmark et al. (2024). The code is written in Python and R. The simulations are computationally intensive and may take several days to run on a standard laptop. The simulations were ran using Python 3.11.3 and R 4.3.2.

## Installing required packages

### Python
We suggest using a virtual environment to avoid conflicts with other Python packages. Run `python -m venv venv` in the terminal to create a virtual environment in the project folder.
Activate the virtual environment by running:
- Windows: `venv\Scripts\activate`
- MacOS/Linux: `source venv/bin/activate`

After activating the virtual environment, install the required packages using `pip install -r requirements.txt`.

### R
To improve reproducibility, the project uses [renv](https://github.com/rstudio/renv). When the project folder is opened in [R-studio](https://posit.co/download/rstudio-desktop/) on your local machine, the required packages with the correct versions will be automatically installed in a local environment, without affecting your existing global R environment. If they are not installed automatically, run `renv::restore()` in the R console to install the required packages.

## Running simulations
- Run `python 1_sample_datasets.py` to sample items and test takers for each simulated scenario.
- Run `python 2_fit_models.py --model "MMC" --learning_rates 0.08 --batch_sizes 64` and `python 2_fit_models.py --model "NR" --learning_rates 0.08 --batch_sizes 128` to fit and evaluate autoencoder estimated MMC and NR models respectively.
    - Note that you can also supply different arguments. For example, run with fewer iterations, or experiment with different learning rates and batch sizes. See `python 2_fit_models.py --help` for details.
- Run `Rscript 2_fit_models.R` to fit and evaluate MML estimated models.
- Run the R script `3_summarize_results.R` to reproduce the article table and figure.

## Notes
As the raw data from the SweSAT dataset cannot be shared due to copyright reasons, a simulated dataset is provided as a replacement to enable the entire simulation process to be mimiced. Thus the results from the simulation study may differ slightly from the results in the article. Additionally, for autoencoder models, the results may also vary due to differences when running python multiprocessing on various operating systems and whether CPU or GPU was used for fittng autoencoder models. However, the overall conclusions should be the same.
