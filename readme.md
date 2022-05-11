# RESISC45Project
## Intro
This is a project using CNNs to perform remote sensing image scene classification on 
RESISC45.
## How to use
* Download RESISC45 dataset, unzip and move it to the same directory.
* Rename it to *Data*.
* Run *gen_dataset.py*.
* Modifying code in *run_net.py* and *metric_learn.py*.
* Run *run_net.py* to train ResNet34 and DenseNet121, run *metric_learn.py* to train ResNet34M and DenseNet121M, or just run *main.py* to train both.