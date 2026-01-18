# citation-score

## python lib

python==3.13.9

```bash
pip install -r requirements.txt
```

## environment

modify `.env.local` name to `.env`

```bash
mv .env.locl .env
```

fill values of key from `.env`

## Data collection

https://www.semanticscholar.org

you can got shcolar data from this website.
you'd better require a api key in order to get data no more limitation.

`database.ipynb`is the code, you can run it create `sample.xlsx`.

## Scoring for relevance and author independence

`score.ipynb`is the code, you can run it create `score.xlsx`.

record cited paper and thier citing paper, then:
1. calculate cross-encoder and cosine similarity from `contexts` and `abstract`
2. calculate cross-encoder from `abstract` both citing and cited to stimulate `contexts` unavailable
3. calculate distance between citing authors and cited authors and translate to [0, 1]