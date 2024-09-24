# Multi-Target Cross-Lingual Summarization

Source code for the EMNLP 2024 (Findings) paper [Multi-Target Cross-Lingual Summarization: a novel task and a language-neutral approach](https://arxiv.org).

## Setup

Create a virtual environment and install the requirements:

```bash
conda create -n mtxlsum python=3.10
conda activate mtxlsum
pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

If you wish to run the evaluation, you will also need to install the multilingual ROUGE  package
available at <https://github.com/csebuetnlp/xl-sum/tree/master/multilingual_rouge_scoring>.

## Data preparation

1. Download the data from the [official repository](https://github.com/csebuetnlp/CrossSum) and extract it to the `original_data` directory.

2. Run the following command to cluster the test data (similarly for validation):

```bash
mkdir -p data
mkdir -p data/test
python aggregate.py --data_dir original_data/test --output_dir data/test --langs arabic chinese_simplified english french portuguese russian spanish
```

This will create a `data` directory with multiple JSONL files. Each line corresponds to a cluster of documents and has the following format:

```
{
    "num_docs": int,
    "url0": str,
    "lang0": str,
    "text0": str,
    "summary0": str,
    "url1": str,
    "lang1": str,
    "text1": str,
    "summary1": str,
    ...
}
```

## Generation

E.g., to generate summaries for all English documents using NeutralRR using all the languages in each cluster as targets, run the following command:

```bash
python generate.py --source_lang=en --split=test --method=rerank --search_mode=dijkstra --num_candidates=8 --temperature=1.0 --top_k=50 --num_sampling_beams=5 --output=predictions_en.jsonl
```

This will create a `predictions_en.jsonl` file where each line has the following format:

```
{
    "source_url": str,
    "summary_english": str,
    "summary_spanish": str,
    ...
}
```

For other methods and options, run `python generate.py --help`.

## Evaluation

To evaluate the generated summaries, run the following command:

```bash
python evaluate.py --predictions=./predictions_en.jsonl --source_lang=en --split=test --output=predictions_en_eval.json
```

This will create a JSON file with the results of the evaluation for each target language according to several metrics.

## Citation

```
@inproceedings{pernes-etal-2024-mtxls,
    title = "Multi-Target Cross-Lingual Summarization: a novel task and a language-neutral approach",
    author = "Pernes, Diogo  and
      Correia, Gon\c{c}alo M.  and
      Mendes, Afonso",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org",
}
```
