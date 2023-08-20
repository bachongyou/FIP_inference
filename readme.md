## run command: 
before run the code, please visit https://github.com/nghorbani/human_body_prior to install the human_body_prior package 

`python evaluate.py`
for evaluation

our data preprocessing is shown as `preprocess.py`

## data download 
before running the code, you may need to download ckpt.zip and data.zip from https://drive.google.com/drive/folders/1rxYTv5j8G-10Sxy1gwmZrP-NUHJPVqyo?usp=drive_link

and unzip them to the folder ./ckpt and ./data respectively。

## data explain:

>_dip_betas.pt_: 
>>the regressed body shape parameters of DIP-IMU objects

>_imu_test.pt_: 
>>the data of preprocessed DIP_IMU raw data, 
including: 'acc' for data of accelerometer, 'ori' for data of gyroscope, 'pose' for gt values.

>SMPL_male.pkl:  the official SMPL model, which can be downloaded on: "SMPL for Python Users" on https://smpl.is.tue.mpg.de/

>_support_data_: the official SMPL model, which can be downloaded on: "DMPLS for AMASS(dmpls)" on https://smpl.is.tue.mpg.de/
> and Extended SMPL+H model(smplh) on https://mano.is.tue.mpg.de
>, release them and place them like below support_data tree.
>
>the folder tree:
>
> support_data
>>body_models
>>>dmpls
>>>
>>>smplh
>>>

**notice** 
The term "gender" used in the dataset and code refers to the same variable as the term "sex" mentioned in the manuscript.

## demo video
for demo video, please refer to https://youtu.be/9jyCHrsTVt8
