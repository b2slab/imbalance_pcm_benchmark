# Introduction

In this study we aim to analyse the effect of different balancing strategies in deep learning proteochemometrics models. The balancing strategies are (1) no resampling, (2) resampling after clustering, (3) resampling before clustering and (4) semi-resampling.

## Material

We are using activity data from Karimi, M., Wu, D., Wang, Z., & Shen, Y. (n.d.). DeepAffinity: Interpretable Deep Learning of Compound-Protein Affinity through 
Unified Recurrent and Convolutional Neural Networks. Retrieved from https://github.com/Shen-Lab/DeepAffinity . 
We focused our study in two protein families: protein kinases (PK) and G-protein coupled receptors (GPCR).

The project is coded in Python 3.6.9.

## Structure

- Functions are defined in the src/ folder (.py files)
- The workflow of the analysis is applied through Jupyter Notebooks in the scripts/ folder. 
The files are preceded by a number that indicates the chronological order of their execution.
- Intermediate and final results are generated into the folder data/.

## Workflow (scripts)

#### 1. Data pre-processing
- 00_data_preprocessing.ipynb
- 01_analysis_protein_families.ipynb
- 02_extracting_protein_info.ipynb
- 03_tuning_padding_type.ipynb

#### 2. Strategies analysis
- **no_resampling**
   - 00_preparing_data.ipynb
   - 01_training_model.ipynb
   - 02_computing_ratios.ipynb

- **resampling_after_clustering**
   - 00_preparing_data.ipynb
   - 01_training_model.ipynb
   - 02_computing_ratios.ipynb
   
- **resampling_before_clustering**
   - 00_preparing_data.ipynb
   - 01_training_model.ipynb
   - 02_computing_ratios.ipynb
   
- **semi_resampling**
   - 00_preparing_data.ipynb
   - 01_training_model.ipynb
   - 02_computing_ratios.ipynb

#### 3. Results processing
- 04_joining_ratios_dfs.ipynb
- 05_creating_random_baseline.ipynb


## System requirements
The runs have been executed on a machine from the B2SLab (Universitat Politecnica de Catalunya) with 12 threads, 32GB RAM and 2x NVIDIA GeForce GTX 1070.

## Considerations
- The absPath variable at the beginning of each notebook and src file should be changed once the repository is cloned to the correct path in each case.
- Execution without GPU of notebooks training_model notebooks may require considerable time and it is not recommended.

