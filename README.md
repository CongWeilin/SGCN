# SGCN

Open-sourced implementation for ICML20 paper (5830) "On the Importance of Sampling in Learning Graph Convolutional Networks: 
Convergence Analysis and Variance Reduction".

## Setup
This implementation is based on [PyTorch >= 1.0.0](https://pytorch.org/) We assume that you're using Python 3 with pip installed. The required dependencies can be installed by executing the following commands.

```bash
# create virtual environment
virtualenv env
source env/bin/activate
# install dependencies 
pip install -r requirements.txt
```
  
Experiments are produced on Pubmed, Flickr, Reddit, and PPi dataset. Dataset can be download from [Google drive](https://drive.google.com/drive/folders/1wW8JwNkPbXZuv1gD4E3BAQ3CHBdqtEqm?usp=sharing).

## Usage
Execute the following scripts (`run.sh`) to reproduce the result in the paper. 

```bash
# create folders that save experiment results and datasets
mkdir ./results
mkdir ./data # please download the dataset and put them inside this folder

# train LADIES
python train_sgcn.py --dataset ppi --sample_method ladies
python train_sgcn.py --dataset flickr --sample_method ladies
python train_sgcn.py --dataset pubmed --sample_method ladies
python train_sgcn.py --dataset reddit --sample_method ladies

# train GraphSage
python train_sgcn.py --dataset ppi --sample_method graphsage
python train_sgcn.py --dataset flickr --sample_method graphsage
python train_sgcn.py --dataset pubmed --sample_method graphsage
python train_sgcn.py --dataset reddit --sample_method graphsage

# train FastGCN
python train_sgcn.py --dataset ppi --sample_method fastgcn
python train_sgcn.py --dataset flickr --sample_method fastgcn
python train_sgcn.py --dataset pubmed --sample_method fastgcn
python train_sgcn.py --dataset reddit --sample_method fastgcn

# train Exact
python train_sgcn.py --dataset ppi --sample_method exact
python train_sgcn.py --dataset flickr --sample_method exact
python train_sgcn.py --dataset pubmed --sample_method exact
python train_sgcn.py --dataset reddit --sample_method exact
```
There are also other hyperparameters to be tuned, which can be found in `train_sgcn.py` for details. 

For a better visualization purpose, we also provide a jupyter notebook file.
Please check `example.ipynb` for details. Notice that `example.ipynb` requires a GPU for calculating the CUDA utilization during training.

## Algorithms
- "sgcn_pplus_0" refers to the zeroth-order only, a.k.a. SGCN++ (zeroth order)
- "sgcn_pplus_1" refers to the first-order only, a.k.a. SGCN++ (first order)
- "sgcn_pplus_01" refers to the doubly variance reduction, a.k.a. SGCN++ (doubly)

