# PAN
Path-Based Attention Neural Model for Fine-Grained Entity Typing 

This repository contains the source code for the experiments presented in the following research paper ([PDF](http://zhangdenghui.tech/docs/PAN.pdf)):

Path-Based Attention Neural Model for Fine-Grained Entity Typing 
Denghui Zhang, Pengshan Cai, Yantao Jia, Manling Li, Yuanzhuo Wang 
The Thirty-Second AAAI Conference on Artificial Intelligence (AAAI 2018) (poster, submitted) 


## Requirements
Pytorch 0.2.0

## Preprocessing

To download and preprocess the corpora, run the following command:
```bash
$ ./preprocess.sh
```
To run the experiements, run command like following:
```bash
$ python train_PAN.py gillick attentive --feature --hier --bags -batch_size 1000 -bag_size 20
```