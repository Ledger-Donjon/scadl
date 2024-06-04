# Tutorial

## Datasets

Scadl uses two different datasets for its tutorial. The first dataset is collected by running a non-protected AES on [ChipWhisperer-Lite](https://rtfm.newae.com/Targets/CW303%20Arm/). The figure shown below indicates the power consumption of the first round AES (top). The bottom figure shows the SNR of **sbox[P^K]**. The yellow zone indicates P^K and the gray zone is related to **sbox[P^K]** of the 16 bytes. The profiling and non-profiling tutorials use the first peak in the gray zone which is related to **sbox[P[0] ^ K[0]]**. The multi-label tutorial uses the first two peaks of **sbox[P[0] ^ K[0]]** and **sbox[P[1] ^ K[1]]**.


![Test Image 6](images/cw_aes.png)


The second dataset is [ASCAD](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1) which is widely used in the side-channel attacks (SCAs) domain.

##  Labeling
we consider for all the experiments, one or several AES Sbox for labeling the DL architectures
```python 
def  leakage_model(metadata):
"""leakage model for sbox[0]"""
return  sbox[metadata["plaintext"][0] ^ metadata["key"][0]]
```
 ## DL models
 For our experiments, we use CNN and MLP models which are the most used DL models by the SCA community.

