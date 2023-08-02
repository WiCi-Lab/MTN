# Multi-Task Learning Based Joint Channel Estimation for Hybrid-Field STAR-RIS Systems
Authors: Jian Xiao, Ji Wang, Zhaolin Wang, Jun Wang, Wenwu Xie, and Yuanwei Liu.

This work has been submitted for possible publication. We highly respect reproducible research, so we try to provide the simulation codes for our submitted papers.

## Usage
How to use this simulation code package?

### 1.Data Generation and Download

In the following link, we have provided the paired samples for hybrid-field cascaded channel estimation in STAR-RIS systems, in which the data preprocessing and normalization operations have been completed.
DOI Link: https://dx.doi.org/10.21227/5v5m-wd56

The simulation parameters of this dataset have been elaborated in our submitted paper. For instance, M_1 x M_2 = 4 x 8, N_1 x N_2 = 4 x 32, f_c = 73GHz, and Q=N/4.  The  description of each data file is listed as follows.

inHmix_28_32_128_K2_32pilot.mat: the training dataset and validation dataset in the ES protocol.<br/>
inHmix_28_32_128_test_K2_32pilot.mat: the testing dataset in the ES protocol.<br/>
inHmix_28_32_128_S_32pilot.mat: the training dataset and validation dataset in the TS protocol.<br/>
inHmix_28_32_128_test_S_32pilot.mat: the testing dataset in the TS protocol.<br/>
please download the dataset and put it in the desired folder, in which the dataset in the TS protocol is prepared for the STN model.

### 2. Training and Testing of MTN/STN model

We have provided the model training and test code to reproduce the corresponding results of this submitted paper. The deatailed description of each script is listed as follows.

main.py: the main function of the MTN model.<br/>
main_STN.py: the main function of the STN model.<br/>
MTN.py: the proposed multi-task network architecture.<br/>
STN.py: the proposed single-task network architecture.<br/>
MDSR.py: a baseline network architecture based on the MDSR model.<br/>
DRSN.py: a baseline network architecture based on the DRSN model.<br/>
Benchmarks.py: A preliminary version based on the Transformer model.<br/>

Specifically, you can run the “main.py” or "main_STN.py" file to train the hybrid-field cascaded channel estimation network, and then test the channel estimation performance under different SNR conditions, in which the model training and test code have been integrated into the main function. To enhance the readability of the provided scripts, We also have added the necessary code annotation.

## Notes 

(1)	Please confirm the required library files have been installed.

(2)	Please switch the desired data loading path and network models.

(3) In the training stage, the different hyper-parameters setup will result in slight difference for final channel estimation perfromance. According to our training experiences and some carried attempts, the hyper-parameters and network architecture can be further optimized to obtain better channel estimation performance gain, e.g., the dividing ratio between training samples and vadilation samples, the number of convolutional kernel, the training learning rate, batchsize and epochs.

(4) Since the limitation of sample space (e.g., the fixed number of channel samples is collected for each user), the inevitable overfitting phenomenon may occur in the network training stage with the increase of epochs

(5) The author in charge of this simulation code pacakge is: Jian Xiao (email: jianx@mails.ccnu.edu.cn). If you have any queries, please don’t hesitate to contact me.

## Acknowledgements

(1) In the hybrid-field channel modeling for STAR-RIS systems, we refer to the channel modeling scheme in the following reference paper for RIS-aided mmWave Massive MIMO systems (e.g., the path loss model and clustered scatters distribution), in which the far-field communication scenarios is extend to the hybrid-field communication by supplementing the near-field array response and the VR cover vector. We are very grateful for the author of the following reference paper and the open-source SimRIS Channel Simulator MATLAB package.<br/>
[1] E. Basar, I. Yildirim, and F. Kilinc, “Indoor and outdoor physical channel modeling and efficient positioning for reconfigurable intelligent surfaces in mmWave bands,” IEEE Trans. Commun., vol. 69, no. 12, pp. 8600-8611, Dec. 2021.<br/>
[2] E. Basar, I. Yildirim, “Reconfigurable Intelligent Surfaces for Future Wireless Networks: A Channel Modeling Perspective“, IEEE Wireless Commun., vol. 28, no. 3, pp. 108–114, June 2021.<br/>

(2) We also appreciate the authors in the following reference paper to provide the open-source code for near-field ELAA channel estimation and STAR-RIS channel estimation, which are refered to the benchmarks in this work.<br/>
[3] Y. Lu and L. Dai, “Near-field channel estimation in mixed LoS/NLoS environments for extremely large-scale MIMO systems,” IEEE Trans. Commun., vol. 71, no. 6, pp. 3694 - 3707, Jun. 2023.<br/>
[4] C. Wu, C. You, Y. Liu, X. Gu, and Y. Cai, “Channel estimation for STAR-RIS-aided wireless communication,” IEEE Commun.Lett., vol. 26, no. 3, pp. 652–656, Mar. 2022.

Copyright reserved by the WiCi Lab, Department of Electronics and Information Engineering, Central China Normal University, Wuhan 430079, China.
