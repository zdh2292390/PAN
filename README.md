# PAN
Path-Based Attention Neural Model for Fine-Grained Entity Typing 

This repository contains the source code for the experiments presented in the following research paper ([PDF](http://zhangdenghui.tech/docs/PAN.pdf)):

Path-Based Attention Neural Model for Fine-Grained Entity Typing 
Denghui Zhang, Manling Li, Pengshan Cai, Yantao Jia, Yuanzhuo Wang, Xueqi Cheng 
The Thirty-Second AAAI Conference on Artificial Intelligence (AAAI 2018) (poster, accepted) 


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

Please cite our paper if you find our research useful or relevant.
```bash
@article{zhang2017path,
  title={Path-based attention neural model for fine-grained entity typing},
  author={Zhang, Denghui and Cai, Pengshan and Jia, Yantao and Li, Manling and Wang, Yuanzhuo and Cheng, Xueqi},
  journal={arXiv preprint arXiv:1710.10585},
  year={2017}
}
```
