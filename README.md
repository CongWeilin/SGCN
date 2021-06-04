# SGCN

Open-sourced implementation for "On the Importance of Sampling in Learning Graph Convolutional Networks: 
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
  
Experiments are produced on Pubmed, Flickr, Reddit, and PPi dataset. Dataset can be download from [Google drive](https://drive.google.com/drive/folders/15eP7OHiHQUnDrHKYh1YPxXkiqGoJhbis) and located inside `./data/` folder.

## Usage
Execute the following scripts (`run_experiments.sh`) to reproduce the result in the paper. 

```bash
python run_experiments.py --sample_method 'ladies' --dataset 'reddit' 
python run_experiments.py --sample_method 'fastgcn' --dataset 'reddit' 
python run_experiments.py --sample_method 'graphsage' --dataset 'reddit' 
python run_experiments.py --sample_method 'vrgcn' --dataset 'reddit' 
python run_experiments.py --sample_method 'graphsaint' --dataset 'reddit' 
python run_experiments.py --sample_method 'exact' --dataset 'reddit' 

python run_experiments.py --sample_method 'ladies' --dataset 'ppi'
python run_experiments.py --sample_method 'fastgcn' --dataset 'ppi'
python run_experiments.py --sample_method 'graphsage' --dataset 'ppi'
python run_experiments.py --sample_method 'vrgcn' --dataset 'ppi'
python run_experiments.py --sample_method 'graphsaint' --dataset 'ppi'
python run_experiments.py --sample_method 'exact' --dataset 'ppi'

python run_experiments.py --sample_method 'ladies' --dataset 'flickr'
python run_experiments.py --sample_method 'fastgcn' --dataset 'flickr'
python run_experiments.py --sample_method 'graphsage' --dataset 'flickr'
python run_experiments.py --sample_method 'vrgcn' --dataset 'flickr'
python run_experiments.py --sample_method 'graphsaint' --dataset 'flickr'
python run_experiments.py --sample_method 'exact' --dataset 'flickr'

python run_experiments.py --sample_method 'ladies' --dataset 'ppi-large'
python run_experiments.py --sample_method 'fastgcn' --dataset 'ppi-large'
python run_experiments.py --sample_method 'graphsage' --dataset 'ppi-large'
python run_experiments.py --sample_method 'vrgcn' --dataset 'ppi-large'
python run_experiments.py --sample_method 'graphsaint' --dataset 'ppi-large'
python run_experiments.py --sample_method 'exact' --dataset 'ppi-large'

python run_experiments.py --sample_method 'ladies' --dataset 'yelp'
python run_experiments.py --sample_method 'fastgcn' --dataset 'yelp'
python run_experiments.py --sample_method 'graphsage' --dataset 'yelp'
python run_experiments.py --sample_method 'vrgcn' --dataset 'yelp'
python run_experiments.py --sample_method 'graphsaint' --dataset 'yelp'
python run_experiments.py --sample_method 'exact' --dataset 'yelp'
```

For a better visualization purpose, we also provide a jupyter notebook file.
Please check `run_experiments.ipynb` for details. Notice that `run_experiments.ipynb` requires a GPU for calculating the CUDA utilization during training.

