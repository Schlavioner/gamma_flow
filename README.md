## Name
GAMMA_FLOW: **G**uided **A**nalysis of **M**ulti-label spectra by **Ma**trix **F**actorization for **L**ightweight **O**perational **W**orkflows

## Description
In this project, a model is trained to analyze gamma spectra. It includes the preprocessing of spectra and training of a dimensionality reduction model. As a result, (unlabelled) test spectra can be denoised and classified, including a classification uncertainty and a measure for the outlier probability. The model can be applied to single-label spectra as well as to multi-label spectra (linear combinations of the training spectra).

## Installation

To execute the jupyter notebooks, it is necessary to have python (ideally >=3.12) and an IDE like Visual Studio Code 
installed. Additionally, it is recommend to have git available for cloning the repository.

*If this is not possible with your setup, you can simply download the repository as zip-file.*

GAMMA_FLOW is generally designed for usage with **Ubuntu** (installation script and instructions below). 
We have successfully tested GAMMA_FLOW on the following operating systems:
- Ubuntu 22.04.5 LTS
- MS Windows 11 Pro
- macOS Catalina 10.15.7

Windows users can try to follow the windows part of installation if winget is available on their machine.
As alternative, you can try to use the linux subsystem of your Windows installation 
(https://learn.microsoft.com/de-de/windows/wsl/install). Then you can use the linux script.

### *WINDOWS*

### Step 1: Install prerequisites

```bash
winget install --id=Microsoft.VisualStudioCode -e

(optional)
winget install -e --id Python.Python.3.12

(optional)
winget install -e --id Git.Git
(if you cannot install, download it manually from https://git-scm.com/downloads/win)

(optional, if you encounter problems with install.bat) 
winget install --id=Anaconda.Anaconda3 -e
```

### Step 2: Clone or download repository

*Cloning* 

> ```bash
> git clone https://gitlab.opencode.de/uba-ki-lab/gamma_flow.git
> ```

*Download* 

Simple download the zip-file from the repository page.

### Step 3: Change into folder gamma_flow
> 
> ```bash
> cd gamma_flow
> ```

or go into the folder.


### Step 4: Execute installation script
> 
> ```bash
> install.bat
> ```

or double-click to start installation.


### *UBUNTU*

### Step 1: Install prerequisites

Ubuntu users can use `snap`. Git should be already installed.

```bash
sudo snap install --classic code
```

### Step 2: Clone or download repository

> ```bash
> git clone https://gitlab.opencode.de/uba-ki-lab/gamma_flow.git
> ```


### Step 3: Change into folder gamma_flow
> 
> ```bash
> cd gamma_flow
> ```

### Step 4: Execute installation script
> 
> ```bash
> sudo chmod +x install.sh
> ./install.sh
> source .venv/bin/activate
> ```

## Usage: Run the minimal example with the provided data

### Step 0: Preparation 

- Open an IDE that can handle both python files and jupyter notebooks (recommendation: VSCode): 
Right-click on the project folder, select `open with` and `Visual Studio Code`
- Install jupyter kernel
- Select the venv inside the notebook (the symbol is comparable to a document shredder)

### Step 1: Preprocessing

- Run all preprocessing steps for the example dataset in [preprocessing.ipynb](preprocessing.ipynb). 
- After this, all preprocessed data should be saved in the folder `data/numpy_ready`.

### Step 2: Train and Test dimensionality reduction
 
Train and test the dimensionality reduction model in [model.ipynb](model.ipynb).
The trained model should be saved in the folder `trained_models` as `trained_dim_model.npy`. 

### Step 3: Explorating outlier detection
 
Explore the options for outlier detection in [outlier.ipynb](outlier.ipynb) and decide for your 
measurement routine which quantity works for you as measure for the probability of a spectrum to be an outlier. 

### Step 4: Usage of trained models
 
Use the trained models for new, unknown measurements with your spectrometer in your own measurement routine. 

## Support
Installation: Oesen, Benjamin <Benjamin.Oesen -AT- uba.de>

Data Science: Rädle, Viola <Viola.Raedle -AT- uba.de>

Project: Hartwig, Tilman <Tilman.Hartwig -AT- uba.de>

Anything else: ki-anwendungslabor -AT- uba.de


## Authors and Acknowledgment
Viola Rädle, Tilman Hartwig, Benjamin Oesen, Julius Vogt, Eike Gericke, Emily Alice Kröger, Martin Baron


## Folder structure
. <br>
├── 01_preprocessing.ipynb &emsp; &emsp; &emsp;  &emsp; &emsp; &emsp; # preprocessing of spectra <br>
├── 02_model.ipynb &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; # model training and testing <br>
├── 03_outlier.ipynb&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp;# notebook for better understanding of outlier detection <br>
├── CONTRIBUTING &emsp; &emsp; &emsp;  &emsp; &emsp; &emsp; # How to contribute to this software <br>
├── copyright notes and licenses.third-parties &emsp; &emsp; &emsp;  &emsp; &emsp; &emsp; # notes concerning imported packages <br>
├── example_dataset.zip &emsp; &emsp; &emsp;  &emsp; &emsp; &emsp; # example data for minimum example <br>
├── install.bat &emsp; &emsp; &emsp;&emsp; &emsp; &emsp;  &emsp; &emsp; &emsp; # installation script for windows <br>
├── install.sh &emsp; &emsp; &emsp;&emsp; &emsp; &emsp;  &emsp; &emsp; &emsp; # installation script for linux <br>
├── LICENSE &emsp; &emsp; &emsp;&emsp; &emsp; &emsp;  &emsp; &emsp; &emsp; # software license for gamma_flow <br>
├── publiccode.yml &emsp; &emsp; &emsp;  &emsp; &emsp; &emsp; # metadata description standard for public software <br>
├── README                              <br>
├── requirements.txt &emsp; &emsp; &emsp;  &emsp; &emsp; &emsp; # all required third-party packages <br>
├── documentation <br>
│   ├── 01_preprocessing.pdf &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; # documentation of preprocessing (for example dataset) <br>
│   ├── 02_model.pdf &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; # documentation of model training (for example dataset) <br>
│   ├── 03_outlier.pdf &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; # documentation of outlier analysis (for example dataset) <br>
│   └── HTML-documentation &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; # code documentation as HTML
├── plots &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp;# folder for saved plots (contains 1 folder per nuclide) <br>
│   ├── Am241 &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; # folder for saved plots of Am241 <br>
│   └── ... <br> 
├── tools &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp;# folder for python files <br> 
│   ├── globals.py &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; # global variables <br> 
│   ├── util.py &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp;# utility functions used by all other python files / notebooks <br>
│   ├── plotting.py &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; # plotting routines <br>
│   ├── tools_preprocessing.py &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; # functions used by 01_preprocessing.ipynb <br>
│   ├── tools_model.py &emsp; &emsp; &emsp;  &emsp; &emsp; &emsp; &emsp; &emsp; &emsp;# functions used by 02_model.ipynb <br> 
│   └── tools_outlier.py &emsp; &emsp; &emsp;  &emsp; &emsp; &emsp; &emsp; &emsp; &emsp;# functions used by 03_outlier.ipynb <br> 
├── trained_models &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; # folder for trained model and calibration spectra <br>
│   └── trained_dim_model.npy &emsp; &emsp; &emsp; &emsp;  # transformation matrix / loadings <br>



## How to use the Project

This project consists of several jupyter notebooks that rely on functions in corresponding python files. In addition, the python files `globals` and `util` provide global variables and basic functions that are used by all notebooks. 
The data should be analyzed in a certain order, as described below: 

**01_preprocessing.ipynb (uses tools_preprocessing.py):** 
- preprocessing of the spectral data: 
    - reads the spectra as lists of dictionaries (format: .npy)
    - rebins spectra to a standard energy calibration 
    - aggregates the spectral data by isotopes and detectors
    - optional: limits the spectra per isotope to a maximum number
- data exploration: 
    - visualizes of mean spectra from different detectors
    - visualizes of example spectra for all isotopes
    - calculates & visualizes a cosine similarity matrix between all isotopes and detectors

**02_model.ipynb (uses tools_model.py):**
- specifies training and validation data
- trains the dimensionality reduction model (build the loadings / transformation matrix)
- applies the model (fit spectra to loadings, result: scores / latent space representation of the spectra)
- classifies validation data 
- visualizes the classification results: 
    - confusion matrix
    - misclassified spectra
    - denoised example spectrum
    - misclassification statistics
    - scores as scatter matrix
    - mean scores as bar plot
 - applies the model to single-label spectra from a detector not used in model training
 - applies the model to multi-label spectra from a detector not used in model training

**03_outlier_analysis.ipynb (uses tools_outlier.py):**
- Exploration of outlier analysis with 3 different ways to identify unknown spectra
- Amongst the known isotopes, we simulate an outlier by pretending that an isotope in unknown and retrain the model based on the remaining known isotopes
- The spectra from the unknown isotope (which was not used for training) can then be used as example outlier
    - Option 1: Decision Trees
    - Option 2: Logistic Regression on most important feature
    - Option 3: Set manual threshold for most important feature
- Results of this notebook can then be manually implemented in measurement pipeline


## List of model parameters
- isotopes to be analyzed (defined manually in `data/00_list_of_isotopes.txt`)
- `dets_measures` and `det_simulated`: names of simulated / measured detectors (`01_preprocessing.ipynb`)
- `min_counts` and `max_counts`: minimum and maximum number of counts allowed per spectra (`01_preprocessing.ipynb`)
- `std_calib`: standard energy calibration for rebinning (`01_preprocessing.ipynb`)
- `n_max`: maximum number of spectra per isotope (`01_preprocessing.ipynb`)
- `dets_tr`: detectors used for model training (`02_model.ipynb`)
- `min_channel`: minimum channel for model training (`02_model.ipynb`)
- `min_scores_norm`: minimum (normalized) score for prediction (`02_model.ipynb`)


