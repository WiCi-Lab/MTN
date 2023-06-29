# Multi-Task Learning Based Joint Channel Estimation for Hybrid-Field STAR-RIS Systems
Authors: Jian Xiao, Ji Wang, Zhaolin Wang, Jun Wang, Wenwu Xie, and Yuanwei Liu.

This work has been submitted for possible publication. We highly respect reproducible research, so we try to provide the simulation codes for our submitted papers.

## Usage
How to use this simulation code package?

### 1.Data Generation and Download

We have provided the paired samples in the following link.
[DOI Link: https://dx.doi.org/10.21227/3c2t-dz81
](https://ieee-dataport.org/documents/star-risce)

You can download the dataset and put it in the desired folder. The “LS_64_256R_6users_32pilot.mat” file includes the training and validation dataset, while the “LS_64_256R_test_6users_32pilot” file is used in the test phase.

### 2.The Training and Testing of LPAN/LPAN-L model

We have integrated the model training and test code, and you can run the “main.py” file to obtain the channel estimation result of the LPAN or LPAN-L model. The detailed network model is given in the “LPAN.py” and “LPAN-L.py”.

## Notes 

(1)	Please confirm the required library files have been installed.

(2)	Please switch the desired data loading path and network models.

(3)	In this work, our goal is to propose a general multi-scale channel estimation network backbone for RIS-aided communication systems. In the model training phase, we did not carefully find the optimal hyper-parameters. Intuitively, hyper-parameters can be further optimized to obtain better channel estimation performance gain, e.g., the training batchsize, epochs, and the depth and width of neural network.

The author in charge of this simulation code pacakge is: Jian Xiao (email: jianx@mails.ccnu.edu.cn). If you have any queries, please don’t hesitate to contact me.

Copyright reserved by the WiCi Lab, Department of Electronics and Information Engineering, Central China Normal University, Wuhan 430079, China.

## Acknowledgements

## Preliminary version
A preliminary version has been uploaded, while the clean code and the related instruction will be updated soon. We have provided the paired samples in the following link.

[DOI Link: https://dx.doi.org/10.21227/3c2t-dz81
](https://ieee-dataport.org/documents/star-risce)
