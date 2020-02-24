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