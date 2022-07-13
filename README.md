# LightningMatrixFactorization
A simple demonstration of a matrix factorization recommender system built with PyTorch and PyTorch Lightning.

# Getting Started

You can set up a virtualenv environment for runnning this project with the following commands, or use equivalent commands for your favorite Python environment manager.

```
virtualenv venv/lmf --python=python3.8
source venv/lmf/bin/activate
pip install torch numpy pandas pytorch-lightning

# you'll also need jupyter if you want to work with the demo notebooks
pip install jupyter

git clone git@github.com:riveSunder/LightningMatrixFactorization.git
cd LightningMatrixFactorization
pip install -e .
```

Next you'll need to download an appropriate dataset. For example, the goodbooks-10k dataset CC BY SA 4.0 [Zygmunt Zając](https://github.com/zygmuntz) can be downloaded with the following `wget` command:


```
wget https://github.com/zygmuntz/goodbooks-10k/blob/master/ratings.csv?raw=true -O data/ratings.csv 
```

Head on over to the `notebooks` directory if you'd like to try the demo in notebook format. Or you can begin a training run from the command line:

```
python -m lmf.train 
``` 
