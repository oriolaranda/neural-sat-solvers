# Generalization of Neural SAT Solvers Through Neural Algorithmic Reasoning

Master's thesis performed at Machine Learning group at TUM, under supervision of Prof. Stephan Günnemann and Johanna Sommer.


## Installation

#### CUDA 11.6.1
```
conda install -c "nvidia/label/cuda-11.6.1" cuda-toolkit
```
#### PyTorch 1.12.0
TODO: Is it really needed cudatoolkit=11.6???
```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
```
#### PyTorch Geometric (including torch-scatter and torch-sparse)
```
conda install pyg -c pyg
```
