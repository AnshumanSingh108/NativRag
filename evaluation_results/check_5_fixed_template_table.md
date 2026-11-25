## RAG QA results on 15 test samples (KB:qa, model: nativrag-finetuned, top-k: 5, BERTScore: bert-base-multilingual-cased).

| Country | Language | N | BERT-P | BERT-R | BERT-F1 | BLEU | ROUGE-1 |
|---|---|---|---|---|---|---|---|
| Bangladesh | English | 5 | 0.7356 | 0.6948 | 0.7136 | 0.0346 | 0.2826 |
| India | Hindi | 5 | 0.7371 | 0.7775 | 0.7553 | 0.2183 | 0.3386 |
| Qatar | English | 5 | 0.7565 | 0.7111 | 0.7328 | 0.0393 | 0.3078 |
| Overall | - | 15 | 0.7431 | 0.7278 | 0.7339 | 0.0974 | 0.3097 |

### Overall, by retrieval of the reference answer

| Reference answer | N | BERT-P | BERT-R | BERT-F1 | BLEU | ROUGE-1 |
|---|---|---|---|---|---|---|
| In retrieved context | 3 | 0.7812 | 0.8330 | 0.8040 | 0.3660 | 0.5193 |
| Not in retrieved context | 12 | 0.7335 | 0.7015 | 0.7164 | 0.0302 | 0.2573 |
