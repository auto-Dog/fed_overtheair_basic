# Federated-Learning (PyTorch)

Implementation of the vanilla federated learning paper: [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629), **with channel simulation for over-the-air computation**


Experiments are produced on MNIST, Fashion MNIST and CIFAR10 (both IID and non-IID). In case of non-IID, the data amongst the users can be split equally or unequally.

Since the purpose of these experiments are to illustrate the effectiveness of the federated learning paradigm, only simple models such as MLP and CNN are used.

## Requirments
Install all the packages from requirments.txt
* Python3
* Pytorch
* Torchvision

## Data
* Download train and test datasets manually or they will be automatically downloaded from torchvision datasets.
* Experiments are run on Mnist, Fashion Mnist and Cifar.
* To use your own dataset: Move your dataset to data directory and write a wrapper on pytorch dataset class.

## Running the experiments
The baseline experiment trains the model in the conventional way.

* To run the baseline experiment with MNIST on MLP using CPU:
```
python src/baseline_main.py --model=mlp --dataset=mnist --epochs=10 --channel
```
* Or to run it on GPU (eg: if gpu:0 is available):
```
python src/baseline_main.py --model=mlp --dataset=mnist --gpu=0 --epochs=10 --channel
```
-----
