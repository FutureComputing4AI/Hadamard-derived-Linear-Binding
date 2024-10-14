<h2 align="center">Hadamard-derived linear Binding (HLB)</h2>

<p align="justify">
Vector Symbolic Architectures (VSAs) are one approach to developing Neuro-symbolic AI, where two vectors in $\mathbb{R}^d$ are 'bound' together to produce a new vector in the same space. VSAs support the commutativity and associativity of this binding operation, along with an inverse operation, allowing one to construct symbolic-style manipulations over real-valued vectors. Most VSAs were developed before deep learning and automatic differentiation became popular and instead focused on efficacy in hand-designed systems. In this work, we introduce the Hadamard-derived linear Binding (HLB), which is designed to have favorable computational efficiency, efficacy in classic VSA tasks, and perform well in differentiable systems.
</p>

### Requirements

CSPS

```properties
conda create --name csps python=3.9 -y && conda activate csps
```

- [PyTorch](https://pytorch.org/get-started/locally/) v1.13.1+cu116

XML

```properties
conda create --name xml python=3.9 -y && conda activate xml
```

- [PyTorch](https://pytorch.org/get-started/locally/) v2.0.1+cu118
- [Pyxclib](https://github.com/kunaldahiya/pyxclib)
- ```pip install cython==3.0.10```
- ```pip install tabulate ```

Classical VSA Tasks

- [Torchhd](https://torchhd.readthedocs.io/en/stable/) ```pip install torch-hd```

### Datasets

* CSPS
    - MNIST, SVHN, CIFAR10, and CIFAR100 are the standard datasets that come with PyTorch library. MiniImageNet can be
      downloaded from [Kaggle](https://www.kaggle.com/datasets/arjunashok33/miniimagenet).
* XML
    - For XML experiments pre-processed features are used. All the datasets can be downloaded from the
      benchmark [website](manikvarma.org/downloads/XC/XMLRepository.html).

### Code

The code is organized in two folders. CSPS code is in the ```CSPS Exp/``` folder and XML code is in the ```XML Exp/```
folder. For both of them, individual folders are used by the name of the dataset. For example, network, and training
files related to CIFAR10 datasets are the ```cifar10/``` subfolder, and so on. We have compared the proposed HLB method
with HRR, VTB, and MAP vector symbolic architectures. Results of HRR are taken from previous papers. Our training code
file name for VTB, MAP, and HLB methods are named as $train_{method name}.py$

Similarly, the ```XML Exp/``` folder contains subfolders for each dataset containing training files. Once again, the
training files are named as $train_{method name}.py$ Two types of data loader are used. ```dataset_fast.py``` is a fast
dataloader for smaller datasets (Bibtex, Mediamill, Delicious). ```dataset_sparse.py``` is a dataloader for loading
larger data files. Code regarding the classical VSA tasks is in the ```Classical VSA Tasks/``` folder.
