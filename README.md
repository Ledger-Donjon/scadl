# SCADL

Following the current direction in Deep Learning (DL), more
recent papers have started to pay attention to the efficiency of DL in
breaking cryptographic implementations.

Scadl is a **side-channel attack tool based on deep learning**. It implements most of the state-of-the-art techniques. 

This project has been developed within the research activities of the   Donjon team (Ledger's security team), to help us during side-channel evaluations.
## Features

Scadl implements the following attacks which have been published before:
 - Normal profiling: A straightforward profiling technique as the attacker will use a known-key dataset to train a DL model. Then, this model is used to attack the unknown-key data set. This technique was presented by the following work: [1](https://eprint.iacr.org/2016/921), [2](https://eprint.iacr.org/2018/053), and [3](https://eprint.iacr.org/2019/578).
 - [Non-profiling DL](https://tches.iacr.org/index.php/TCHES/article/view/7387:) A similar technique to differential power Analysis ([DPA](https://paulkocher.com/doc/DifferentialPowerAnalysis.pdf)) but it has the several advantages over DPA to attack protected designs (masking and desynchronization).
 - [Multi-label DL](https://eprint.iacr.org/2020/436): A technique to attack multiple keys using only one DL model.    

## Installation
It can be installed using python3

    python setup.py install

## Requirements
- [keras](https://keras.io/)
- [matplotlib](https://matplotlib.org/)
- [numpy](https://numpy.org/)
- [tensorflow](https://www.tensorflow.org/)

## Tutorial
The tutorial folder contains several examples of normal profiling, multi-label profiling, and non-profiling techniques. The current tutorial has been done using [ChipWhisperer-Lite](https://rtfm.newae.com/Targets/CW303%20Arm/) for a non-protected AES.
