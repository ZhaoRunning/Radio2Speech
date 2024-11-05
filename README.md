# Radio2Speech: High Quality Speech Recovery from Radio Frequency Signals
<p align="center"> <img src='examples/overall.png' align="center"> </p>

> [**Radio2Speech: High Quality Speech Recovery from Radio Frequency Signals**](https://arxiv.org/pdf/2206.11066),               
> Running Zhao, Jiangtao Yu, Tingle Li, Hang Zhao, Edith C.H. Ngai   
> *INTERSPEECH 2022 ([https://arxiv.org/pdf/2206.11066](https://arxiv.org/pdf/2206.11066))*    
> *Project page ([https://zhaorunning.github.io/Radio2Speech/](https://zhaorunning.github.io/Radio2Speech/))*
> 

## Requirements
### Feature generation
To do: Add more details for signal preprocessing
- Transform audio and radio signals from 16kHz to 8kHz
- Generate mel-spectrogram with settings in `config/fbank.yaml`
- Normalization 

### Install Parallel WaveGAN
- Install Parallel WaveGAN from [https://github.com/kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)

## Training
Download the pre-trained weights of Transformer layers in the following Google Drivelinks, and change the corresponding path in `config/train.yaml` to your own path:
```
https://drive.google.com/drive/folders/1WT-MsZ8tlJGILGlDCYs0XNv8eUqv71Zt?usp=sharing
```
Change the path in `train.sh` to your own path, then run
```shell script
./train.sh
```

## Evaluation
Download the pre-trained model of our TransUnet and Parallel WaveGAN in the following Google Drivelinks:
```
https://drive.google.com/drive/folders/1WT-MsZ8tlJGILGlDCYs0XNv8eUqv71Zt?usp=sharing
```
    
Change the path in `eval.sh` and `examples/LJSpeech_val.csv` to your own path, then run
```shell script
./eval.sh
```

## Citation

If you find this repo useful for your research, please consider citing our papers:
```
@inproceedings{zhao22i_interspeech,
  title     = {Radio2Speech: High Quality Speech Recovery from Radio Frequency Signals},
  author    = {Running Zhao and Jiangtao Yu and Tingle Li and Hang Zhao and Edith C. H. Ngai},
  year      = {2022},
  booktitle = {Interspeech 2022},
  pages     = {4666--4670},
  doi       = {10.21437/Interspeech.2022-738},
  issn      = {2958-1796},
}
```
```
@article{10.1145/3610873,
author = {Zhao, Running and Yu, Jiangtao and Zhao, Hang and Ngai, Edith C.H.},
title = {Radio2Text: Streaming Speech Recognition Using mmWave Radio Signals},
year = {2023},
issue_date = {September 2023},
address = {New York, NY, USA},
volume = {7},
number = {3},
doi = {10.1145/3610873},
journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
month = sep,
articleno = {142},
numpages = {28}
}
```

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](docs/LICENSE) file for details.
