## run command: 
before run the code, please visit https://github.com/nghorbani/human_body_prior to install the human_body_prior package 

`python evaluate.py`
for evaluation

our data preprocessing is shown as `preprocess.py`

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

## demo video
for demo video, please refer to https://youtu.be/9jyCHrsTVt8
