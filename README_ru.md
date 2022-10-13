# Датасет Golos 

Golos — это датасет для распознавания речи на русском языке. Он состоит из аудиозаписей речи и транскрипций, полученных с помощью ручной разметки на краудсорсинговой платформе. Общая длительность записей составляет примерно 1240 часов. Все данные и обученные на них акустические модели распознавания речи бесплатны и открыты для скачивания. Также доступны триграммные модели KenLM, подготовленные при помощи русских текстов из открытого корпуса Common Crawl.

# Содержание

- [Структура датасета](https://github.com/salute-developers/golos/#Структура-датасета)
- [Скачать](https://github.com/salute-developers/golos/#Скачать)
  - [Аудиофайлы в формате opus](https://github.com/salute-developers/golos/#Аудиофайлы-в-формате-opus)
  - [Аудиофайлы в формате wav](https://github.com/salute-developers/golos/#Аудиофайлы-в-формате-wav)
  - [Акустические и языковые модели](https://github.com/salute-developers/golos/#Акустические-и-языковые-модели)
- [Оценка качества](https://github.com/salute-developers/golos/#Оценка-качества)
- [Полезные ссылки](https://github.com/salute-developers/golos/#Полезные-ссылки)
- [Лицензия](https://github.com/salute-developers/golos/#Лицензия)
- [Контакты](https://github.com/salute-developers/golos/#Контакты)


## **Структура датасета**

| Домен         | Train файлы | Train (часов)  | Test файлы | Test (часов) |
|----------------|------------|--------|-------|------|
| Crowd          | 979 796    | 1 095  | 9 994 | 11.2 |
| Farfield       | 124 003    |   132.4| 1 916 |  1.4 |
| Итого          | 1 103 799  | 1 227.4|11 910 | 12.6 |

---

## **Скачать**

[MD5 контральные суммы](https://github.com/salute-developers/golos/blob/master/md5sum.txt)


### **Аудиофайлы в формате opus**

| Archive          | Size       |  Link               |
|------------------|------------|---------------------|
| golos_opus.tar   | 20.5 GB    | https://sc.link/JpD |

---

### **Аудиофайлы в формате wav**

Файл с транскрипциями записей всего обучающего подмножества доступен в архиве train_crowd9.tar, доступный по ссылке в таблице:

| Archives          | Size       |  Links              |
|-------------------|------------|---------------------|
| train_farfield.tar| 15.4 GB    | https://sc.link/1Z3 |
| train_crowd0.tar  | 11 GB      | https://sc.link/Lrg |
| train_crowd1.tar  | 14 GB      | https://sc.link/MvQ |
| train_crowd2.tar  | 13.2 GB    | https://sc.link/NwL |
| train_crowd3.tar  | 11.6 GB    | https://sc.link/Oxg |
| train_crowd4.tar  | 15.8 GB    | https://sc.link/Pyz |
| train_crowd5.tar  | 13.1 GB    | https://sc.link/Qz7 |
| train_crowd6.tar  | 15.7 GB    | https://sc.link/RAL |
| train_crowd7.tar  | 12.7 GB    | https://sc.link/VG5 |
| train_crowd8.tar  | 12.2 GB    | https://sc.link/WJW |
| train_crowd9.tar  | 8.08 GB    | https://sc.link/XKk |
| test.tar          | 1.3 GB     | https://sc.link/Kqr |

---

### **Акустические и языковые модели**

Акустическая модель на основе архитектуры [QuartzNet15x5](https://arxiv.org/pdf/1910.10261.pdf) обучена с использованием [NeMo toolkit](https://github.com/NVIDIA/NeMo/tree/r1.0.0b4)


Три n-грамные языковые модели (LM) подготовлены с использованием [KenLM Language Model Toolkit](https://kheafield.com/code/kenlm)

* LM на русских текстах корпуса [Common Crawl](https://commoncrawl.org) 
* LM на текстах транскрипций обучающей подвыборки [Golos](https://github.com/salute-developers/golos)
* LM на русских текстах [Common Crawl](https://commoncrawl.org) и транскрипциях [Golos](https://github.com/salute-developers/golos) вместе (50/50)

| Архивы                   | Объем      |  Ссылки          |
|--------------------------|------------|-----------------|
| QuartzNet15x5_golos.nemo | 68 MB      | https://sc.link/ZMv |
| KenLMs.tar               | 4.8 GB     | https://sc.link/YL0  |


Все данные и модели Golos также доступны в хранилище DataHub ML Space. Там распологаются предобученные модели, датасеты и Docker-образы.


## **Оценка качества**

Процент ошибки распознавания WER (Word Error Rate) для разных тестовых данных.


| Декодер \ Тестовые данные  | Crowd test  | Farfield test    | MCV<sup>1</sup> dev | MCV<sup>1</sup> test |
|-------------------------------------|-----------|----------|-----------|----------|
| Greedy decoder                      | 4.389 %   | 14.949 % | 9.314 %   | 11.278 % |
| Beam Search + Common Crawl LM    | 4.709 %   | 12.503 % | 6.341 %   | 7.976 % |
| Beam Search + Golos train set LM | 3.548 %   | 12.384 % |  -        | -       |
| Beam Search + Common Crawl and Golos LM | 3.318 %   | 11.488 % | 6.4 %     | 8.06 %   |


<sup>1</sup> [Common Voice](https://commonvoice.mozilla.org) - проект компании Mozilla по сбору данных для автоматического распознавания речи.

##  **Полезные ссылки**

[[arxiv.org] Golos: Russian Dataset for Speech Research](https://arxiv.org/abs/2106.10161)

[[habr.com] Golos — самый большой русскоязычный речевой датасет, размеченный вручную, теперь в открытом доступе](https://habr.com/ru/company/sberdevices/blog/559496/)

[[habr.com] Как улучшить распознавание русской речи до 3% WER с помощью открытых данных](https://habr.com/ru/company/sberdevices/blog/569082/)

## **Лицензия**

[English Version](https://github.com/salute-developers/golos/blob/master/license/en_us.pdf)

[Russian Version](https://github.com/salute-developers/golos/blob/master/license/ru.pdf)

## **Контакты**

Создавайте GitHub issue!

Авторы а алфавитном порядке:
- Александр Денисенко
- Ангелина Коваленко
- Николай Карпов 
- Федор Минькин
