# Source Code for PCL

### Prototypical Classifier for Robust Class-Imbalanced Learning

**Authors:** Tong Wei, Jiang-Xin Shi, Yu-Feng Li, and Min-Ling Zhang

[[`arXiv`](https://arxiv.org/pdf/2110.11553.pdf)] [slides] [[`bibtex`](#Citation)]

This paper won the **Best Paper Award** at PAKDD 2022!

## Installation

**Requirements**

* Python 3.6
* torchvision 0.9.0
* Pytorch 1.8.0

**Dataset Preparation**
* [CIFAR-10 & CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)


## Training
```
python main.py --exp cifar10_imb
```
```
python main.py --exp cifar100_imb
```

The saved folder (including logs and checkpoints) is organized as follows.
```
PCL
├── experiment
│   ├── cifar10_exp_0.1_imb_0.2
│   │   ├── model_best.pth.tar
│   │   └── logs
│   │       └── modelname.txt
│   ...   
```


## <a name="Citation"></a>Citation

Please consider citing PCL in your publications if it helps your research. :)

```bib
@inproceedings{wei2022pcl,
    title={Prototypical Classifier for Robust Class-Imbalanced Learning},
    author={Tong Wei, Jiang-Xin Shi, Yu-Feng Li, and Min-Ling Zhang},
    booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
    year={2022},
}
```

## Contact

If you have any questions about our work, feel free to contact us through email (Tong Wei: weit@lamda.nju.edu.cn) or Github issues.


