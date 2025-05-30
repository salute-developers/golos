# Golos dataset

Golos is a Russian corpus suitable for speech research. The dataset mainly consists of recorded audio files manually annotated on the crowd-sourcing platform. The total duration of the audio is about 1240 hours. 
We have made the corpus freely available for downloading, along with the acoustic model prepared on this corpus. 
Also we create 3-gram KenLM language model using an open Common Crawl corpus.

# Table of contents

- [Dataset structure](https://github.com/salute-developers/golos/tree/master/golos#dataset-structure)
- [Downloads](https://github.com/salute-developers/golos/tree/master/golos#downloads)
  - [Audio files in opus format](https://github.com/salute-developers/golos/tree/master/golos#audio-files-in-opus-format)
  - [Audio files in wav format](https://github.com/salute-developers/golos/tree/master/golos#audio-files-in-wav-format)
  - [Acoustic and language models](https://github.com/salute-developers/golos/tree/master/golos#acoustic-and-language-models)
- [Evaluation](https://github.com/salute-developers/golos/tree/master/golos#evaluation)
- [Resources](https://github.com/salute-developers/golos/tree/master/golos#resources)
- [License](https://github.com/salute-developers/golos/tree/master/golos#license)
- [Contacts](https://github.com/salute-developers/golos/tree/master/golos#contacts)


## **Dataset structure**

| Domain         | Train files | Train hours  | Test files | Test hours |
|----------------|------------|--------|-------|------|
| Crowd          | 979 796    | 1 095  | 9 994 | 11.2 |
| Farfield       | 124 003    |   132.4| 1 916 |  1.4 |
| Total          | 1 103 799  | 1 227.4|11 910 | 12.6 |

---

## **Downloads**

[MD5 Checksums](https://github.com/salute-developers/golos/blob/master/golos/md5sum.txt)


### **Audio files in opus format**

| Archive          | Size       | Link                                                                                         |
|------------------|------------|----------------------------------------------------------------------------------------------|
| golos_opus.tar   | 20.5 GB    | [golos_opus.tar](https://cdn.chatwm.opensmodel.sberdevices.ru/golos/golos_opus.tar) |

---

### **Audio files in wav format**

Manifest files with all the training transcription texts are in the train_crowd9.tar archive listed in the table:

| Archives          | Size       | Links                                                                                                |
|-------------------|------------|------------------------------------------------------------------------------------------------------|
| train_farfield.tar| 15.4 GB    | [train_farfield.tar](https://cdn.chatwm.opensmodel.sberdevices.ru/golos/train_farfield.tar) |
| train_crowd0.tar  | 11 GB      | [train_crowd0.tar](https://cdn.chatwm.opensmodel.sberdevices.ru/golos/train_crowd0.tar)     |
| train_crowd1.tar  | 14 GB      | [train_crowd1.tar](https://cdn.chatwm.opensmodel.sberdevices.ru/golos/train_crowd1.tar)     |
| train_crowd2.tar  | 13.2 GB    | [train_crowd2.tar](https://cdn.chatwm.opensmodel.sberdevices.ru/golos/train_crowd2.tar)     |
| train_crowd3.tar  | 11.6 GB    | [train_crowd3.tar](https://cdn.chatwm.opensmodel.sberdevices.ru/golos/train_crowd3.tar)     |
| train_crowd4.tar  | 15.8 GB    | [train_crowd4.tar](https://cdn.chatwm.opensmodel.sberdevices.ru/golos/train_crowd4.tar)     |
| train_crowd5.tar  | 13.1 GB    | [train_crowd5.tar](https://cdn.chatwm.opensmodel.sberdevices.ru/golos/train_crowd5.tar)     |
| train_crowd6.tar  | 15.7 GB    | [train_crowd6.tar](https://cdn.chatwm.opensmodel.sberdevices.ru/golos/train_crowd6.tar)     |
| train_crowd7.tar  | 12.7 GB    | [train_crowd7.tar](https://cdn.chatwm.opensmodel.sberdevices.ru/golos/train_crowd7.tar)     |
| train_crowd8.tar  | 12.2 GB    | [train_crowd8.tar](https://cdn.chatwm.opensmodel.sberdevices.ru/golos/train_crowd8.tar)     |
| train_crowd9.tar  | 8.08 GB    | [train_crowd9.tar](https://cdn.chatwm.opensmodel.sberdevices.ru/golos/train_crowd9.tar)     |
| test.tar          | 1.3 GB     | [test.tar](https://cdn.chatwm.opensmodel.sberdevices.ru/golos/test.tar)                     |

---

### **Acoustic and language models**

Acoustic model built using [QuartzNet15x5](https://arxiv.org/pdf/1910.10261.pdf) architecture and trained using [NeMo toolkit](https://github.com/NVIDIA/NeMo/tree/r1.0.0b4)


Three n-gram language models created using [KenLM Language Model Toolkit](https://kheafield.com/code/kenlm)

* LM built on [Common Crawl](https://commoncrawl.org) Russian dataset
* LM built on [Golos](https://github.com/salute-developers/golos) train set
* LM built on [Common Crawl](https://commoncrawl.org) and [Golos](https://github.com/salute-developers/golos/tree/master/golos) datasets together (50/50)

| Archives                 | Size       | Links                                                                                                                 |
|--------------------------|------------|-----------------------------------------------------------------------------------------------------------------------|
| QuartzNet15x5_golos.nemo | 68 MB      | [QuartzNet15x5_golos.nemo (TO BE UPDATED)](https://cdn.chatwm.opensmodel.sberdevices.ru/golos/QuartzNet15x5_golos.nemo)      |
| CitriNet_ru1024bpe.tgz   | 541 MB     | [CitriNet_ru1024bpe.tgz](https://cdn.chatwm.opensmodel.sberdevices.ru/golos/citrinet/CitriNet_ru1024bpe.tgz) |
| KenLMs.tar               | 4.8 GB     | [KenLMs.tar](https://cdn.chatwm.opensmodel.sberdevices.ru/golos/kenlms.tar)                                  |


Golos data and models are also available in the hub of pre-trained models, datasets, and containers - DataHub ML Space. You can train the model and deploy it on the high-performance SberCloud infrastructure in [ML Space](https://cloud.ru/) - full-cycle machine learning development platform for DS-teams collaboration based on the Christofari Supercomputer.


## **Evaluation**

Percents of Word Error Rate for different test sets


| Decoder \ Test set    | Crowd test  | Farfield test    | MCV<sup>1</sup> dev | MCV<sup>1</sup> test |
|-------------------------------------|-----------|----------|-----------|----------|
| Greedy decoder                      | 4.389 %   | 14.949 % | 9.314 %   | 11.278 % |
| Beam Search with Common Crawl LM    | 4.709 %   | 12.503 % | 6.341 %   | 7.976 % |
| Beam Search with Golos train set LM | 3.548 %   | 12.384 % |  -        | -       |
| Beam Search with Common Crawl and Golos LM | 3.318 %   | 11.488 % | 6.4 %     | 8.06 %   |


<sup>1</sup> [Common Voice](https://commonvoice.mozilla.org) - Mozilla's initiative to help teach machines how real people speak.

##  **Resources**

[[INTERSPEECH 2021] Golos: Russian Dataset for Speech Research](https://www.isca-speech.org/archive/pdfs/interspeech_2021/karpov21_interspeech.pdf)

[[habr.com] Golos — самый большой русскоязычный речевой датасет, размеченный вручную, теперь в открытом доступе](https://habr.com/ru/company/sberdevices/blog/559496/)

[[habr.com] Как улучшить распознавание русской речи до 3% WER с помощью открытых данных](https://habr.com/ru/company/sberdevices/blog/569082/)

## **Cite**
Karpov, N., Denisenko, A., Minkin, F. (2021) Golos: Russian Dataset for Speech Research. Proc. Interspeech 2021, 1419-1423, doi: 10.21437/Interspeech.2021-462
```
@inproceedings{karpov21_interspeech,
  author={Nikolay Karpov and Alexander Denisenko and Fedor Minkin},
  title={{Golos: Russian Dataset for Speech Research}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={1419--1423},
  doi={10.21437/Interspeech.2021-462}
}
```

## **License**

[English Version](https://github.com/salute-developers/golos/blob/master/license/en_us.pdf)

[Russian Version](https://github.com/salute-developers/golos/blob/master/license/ru.pdf)

## **Contacts**

Please create a GitHub issue!

Authors (in alphabetic order):
- Alexander Denisenko
- Angelina Kovalenko
- Fedor Minkin
- Nikolay Karpov
