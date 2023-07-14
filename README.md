A sequence to sequence model for machine translation from source language (English) to target language (Russian) using PyTorch library for the most part.
The batching, batch padding,preprocessing and evaluation of the sentences are all written from scratch.

Dataset:
* [English to Russian Language](https://www.kaggle.com/datasets/jayantawasthi/english-to-russian-language)

Evaluation Metrics:
* BLEU
* PERPLEXITY
* METEOR

Some of the results:
| Num. Of Pairs  | BLEU | METEOR | PERPLEXITY | Time |
| ------------- | ------------- |------------- |------------- | ------------- |
| 50K  | 40.23  | 31.45 | 17.13 | 317.29 |
| 100K  | 49.82  | 40.00 | 9.83 | 1513.91 |

This is an example of a translation.
* Console Output
  ```sh
  The given sentense is: it isnt big
  The transtated sentence is: он небольшой
  The prediction is: оно не большой
  ```
