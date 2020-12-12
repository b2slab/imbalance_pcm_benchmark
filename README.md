# Introduction

In this study we aim to XXXXXX

## Material

We are using activity data from Karimi, M., Wu, D., Wang, Z., & Shen, Y. (n.d.). DeepAffinity: Interpretable Deep Learning of Compound-Protein Affinity through 
Unified Recurrent and Convolutional Neural Networks. Retrieved from https://github.com/Shen-Lab/DeepAffinity . 
We focused our study in two protein families: protein kinases (PK) and G-protein coupled receptors (GPCR).

The project is coded in Python 3.6.9.

## Structure

- Functions are defined in the src/ folder (.py files)
- The workflow of the analysis is applied through Jupyter Notebooks in the scripts/ folder. ORDEN?
The files are preceded by a number that indicates the chronological order of their execution.
- Intermediate and final results are generated into the folder data/.

## Workflow (scripts)

FALTA

## System requirements
The runs have been executed on a machine from the B2SLab (Universitat Politecnica de Catalunya) with 12 threads, 32GB RAM and 2x NVIDIA GeForce GTX 1070.

## Considerations
- The absPath variable at the beginning of each notebook and src file should be changed once the repository is cloned to the correct path in each case.
- Execution without GPU of notebooks training_models notebooks may require considerable time and it is not recommended.

