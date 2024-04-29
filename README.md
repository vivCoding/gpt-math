# gpt math

Extension of [this paper](https://arxiv.org/abs/2203.10316). Finetuned `gpt-3.5-turbo-0125` to use a deductive approach when solving math problems.

## Project setup

```bash
pip install -r requirements.txt
```

### Env variables

- `OPENAI_API_KEY`: api key
- `GPT_MODEL`: id of finetuned model

## Datasets

- `data/mawps-steps-train.json` and `data/ft-mawps-train.json` (MAWPs training)
- `data/mawps-steps-dev.json` and `data/ft-mawps-dev.json` (MAWPs validation)
- `data/mawps-dev.json` (MAWPs validation)
- `data/svamp.json` (SVAMP validation)

The first two datasets (`mawps-steps`) were used during training and had numbers masked. They were converted to `ft-mawps` dataset were used for GPT training. The last two datasets did not have numbers masked and were used for the final evaluation.


## Results

Baseline model is vanilla `gpt-3.5-turbo-0125`. Metric is percentage of problems correct.

|            | MAWPs | SVAMP |
| ---------- | ----- | ----- |
| Baseline   | 0.818 | 0.673 |
| Fine-tuned | 0.882 | 0.717 |

### Results based on number of operations required in final answer.

MAWPs dataset:

| # of Steps | Baseline | Fine-tuned |
| ---------- | -------- | ---------- |
| 1          | 0.92     | 0.93       |
| 2          | 0.669    | 0.93       |
| 3          | 0.2      | 0.5        |
| 4          | 0        | 0.5        |

SVAMP dataset:

| # of Steps | Baseline | Fine-tuned |
| ---------- | -------- | ---------- |
| 0          | 0        | 0          |
| 1          | 0.73     | 0.73       |
| 2          | 0.46     | 0.67       |


## Future work

- Try testing with [RobustMath dataset](https://github.com/zhouzihao501/MathAttack)
- Maybe build thing from scratch (BERT and custom architecture on top) ([maybe](https://arxiv.org/abs/2203.10316v4))
    - Look into more search approaches with LLM guidance?