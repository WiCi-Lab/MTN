# Multi-Task Learning Based Joint Channel Estimation for Hybrid-Field STAR-RIS Systems
Authors: Jian Xiao, Ji Wang, Zhaolin Wang, Jun Wang, Wenwu Xie, and Yuanwei Liu.

This work has been submitted for possible publication. We highly respect reproducible research, so we try to provide the simulation codes for our submitted papers.

## Usage
How to use this simulation code package?

### 1.Data Generation and Download

In the following link, we have provided the paired samples for hybrid-field cascaded channel estimation in STAR-RIS systems, in which the data preprocessing and normalization operations have been completed.
[DOI Link: https://dx.doi.org/10.21227/3c2t-dz81
](https://ieee-dataport.org/documents/star-risce)

The simulation parameters of this dataset have been elaborated in our submitted paper. For instance, M_1 x M_2 = 4 x 8, N_1 x N_2 = 4 x 32, f_c = 73GHz, and Q=N/4.

You can download the dataset and put it in the desired folder. The “inHmix_28_32_128_K2_32pilot.mat” file collect the training dataset and validation dataset, while the “inHmix_28_32_128_test_K2_32pilot.mat” file is the testing dataset.

In the training and validation dataset generation, a common set of XL-RIS links is used for cascaded channel samples. However, in the test dataset, we replace a new set of XL-RIS links for each sample realization, which supports the network generalization in the test stage.

2.The Training and Testing of U-MLP model

We have provided the model training and test code to reproduce the corresponding results. Specifically, you can run the “main_UMLP.py” file to train the channel estimation network, and then run the “test_UMLP.py” to realize the cascaded channel estimation under different SNR conditions. The detailed network architecture is given in the “model_UMLP.py”.

### 2.The Training and Testing of LPAN/LPAN-L model

We have integrated the model training and test code, and you can run the “main.py” file to obtain the channel estimation result of the LPAN or LPAN-L model. The detailed network model is given in the “LPAN.py” and “LPAN-L.py”.

## Notes 

(1)	Please confirm the required library files have been installed.

(2)	Please switch the desired data loading path and network models.

(3)	In this work, our goal is to propose a general multi-scale channel estimation network backbone for RIS-aided communication systems. In the model training phase, we did not carefully find the optimal hyper-parameters. Intuitively, hyper-parameters can be further optimized to obtain better channel estimation performance gain, e.g., the training batchsize, epochs, and the depth and width of neural network.

The author in charge of this simulation code pacakge is: Jian Xiao (email: jianx@mails.ccnu.edu.cn). If you have any queries, please don’t hesitate to contact me.

Copyright reserved by the WiCi Lab, Department of Electronics and Information Engineering, Central China Normal University, Wuhan 430079, China.

## Acknowledgements

In the hybrid-field channel modeling for XL-RIS systems, we refer to the channel modeling scheme in [1] for RIS-aided mmWave Massive MIMO systems (e.g., the path loss model and clustered scatters distribution), in which the far-field communication scenarios is extend to the hybrid-field communication by supplementing the near-field array response and VR cover vector. We are very grateful for the author of following reference paper and the open-source SimRIS Channel Simulator MATLAB package [2].

[1] E. Basar, I. Yildirim, and F. Kilinc, “Indoor and outdoor physical channel modeling and efficient positioning for reconfigurable intelligent surfaces in mmWave bands,” IEEE Trans. Commun., vol. 69, no. 12, pp. 8600-8611, Dec. 2021.

[2] E. Basar, I. Yildirim, “Reconfigurable Intelligent Surfaces for Future Wireless Networks: A Channel Modeling Perspective“, IEEE Wireless Commun., vol. 28, no. 3, pp. 108–114, June 2021.

## Preliminary version
A preliminary version has been uploaded, while the clean code and the related instruction will be updated soon. We have provided the paired samples in the following link.

[DOI Link: https://dx.doi.org/10.21227/3c2t-dz81
](https://ieee-dataport.org/documents/star-risce)
