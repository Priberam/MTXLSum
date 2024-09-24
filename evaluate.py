import argparse
import json
import os
import random
import re

import fasttext
import numpy as np
import torch
from comet import download_model, load_from_checkpoint
from comet.models.base import CometModel
from datasets import load_dataset
from fasttext.FastText import _FastText
from iso639 import languages
from rouge_score import rouge_scorer
from scipy.stats import bootstrap
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.models.blaser.loader import load_blaser_model
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

from dataloading import CrossSumAggregated
from models.aux_models import SONARTextEncoder

WHITESPACE_HANDLER = lambda k: re.sub("\s+", " ", re.sub("\n+", " ", k.strip()))


def compute_rouge(
    refs: list[str],
    preds: list[str],
    lang: str,
) -> dict[str, list[float]]:

    rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False, lang=lang
    )
    rouge_scores = [rouge.score(r, p) for r, p in zip(refs, preds)]
    rouge_scores = {
        metric: [score[metric].fmeasure for score in rouge_scores]
        for metric in rouge_scores[0]
    }
    return rouge_scores


@torch.inference_mode()
def compute_sonar_similarities(
    texts_a: list[str],
    texts_b: list[str],
    langs_a: list[str],
    langs_b: list[str],
    sentence_encoder: SONARTextEncoder,
    blaser: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    embs_a = sentence_encoder.predict(texts_a, source_langs=langs_a, batch_size=64)
    embs_b = sentence_encoder.predict(texts_b, source_langs=langs_b, batch_size=64)

    cos_sim = cosine_similarity(embs_a, embs_b)
    blaser.eval()
    blaser_score = blaser(src=embs_a, mt=embs_b).squeeze()
    return cos_sim, blaser_score


@torch.inference_mode()
def compute_comet(
    src: list[str],
    mt: list[str],
    comet: CometModel,
    batch_size: int = 8,
    device: torch.device = torch.device("cuda:0"),
):
    data = [
        {
            "src": src_i,
            "mt": mt_i,
        }
        for src_i, mt_i in zip(src, mt)
    ]
    device = comet.device
    device_id = int(str(device).split(":")[-1])
    outputs = comet.predict(data, batch_size=batch_size, devices=[device_id])
    comet = comet.to(device)
    return outputs.scores  # type: ignore


def compute_ref_metrics(
    refs: list[str],
    preds: list[str],
    lang: str,
    sentence_encoder: SONARTextEncoder,
    blaser: torch.nn.Module,
) -> dict[str, list[float]]:

    outputs = compute_rouge(refs, preds, lang)
    sonar_cos_sim, blaser_score = compute_sonar_similarities(
        refs, preds, [lang] * len(refs), [lang] * len(preds), sentence_encoder, blaser
    )
    outputs["sonar_cos_sim"] = sonar_cos_sim.cpu().numpy().tolist()
    outputs["blaser"] = blaser_score.cpu().numpy().tolist()
    return outputs


@torch.inference_mode()
def compute_consistency_metrics(
    preds: list[list[str]],
    langs: list[list[str]],
    sentence_encoder: SONARTextEncoder,
    blaser: torch.nn.Module,
    comet: CometModel,
    ref_lang: str | None = None,
) -> dict[str, dict[str, list[float]]]:

    blaser.eval()
    preds_flat = [p for ps in preds for p in ps]
    langs_flat = [l for ls in langs for l in ls]
    embs = sentence_encoder.predict(preds_flat, source_langs=langs_flat, batch_size=64)

    outputs = {}

    if ref_lang is None:  # metrics are computed vs all languages and then averaged
        i = 0
        for cluster, langs_cluster in zip(preds, langs):
            if len(cluster) == 1:
                continue

            embs_cluster = embs[i : i + len(cluster)]

            for l in langs_cluster:
                if l not in outputs:
                    outputs[l] = {"blaser": [], "comet": [], "sonar_cos_sim": []}

            sonar_cos_sim = cosine_similarity(
                embs_cluster.unsqueeze(2), embs_cluster.T.unsqueeze(0)
            )
            for j, l in enumerate(langs_cluster):
                embs_j = embs_cluster[j].unsqueeze(0).expand(len(cluster) - 1, -1)
                embs_other = torch.cat(
                    [
                        embs_cluster[k].unsqueeze(0)
                        for k in range(len(cluster))
                        if k != j
                    ]
                )
                blaser_scores_j = blaser(src=embs_other, mt=embs_j).squeeze()
                outputs[l]["blaser"].append(blaser_scores_j.mean().item())
                outputs[l]["sonar_cos_sim"].append(sonar_cos_sim[j].mean().item())

            i += len(cluster)

        preds_flat, langs_flat = None, None
        langs_unique = outputs.keys()

        for l in langs_unique:
            refs4comet = []
            mt4comet = []
            cluster_lens = []
            for cluster, langs_cluster in zip(preds, langs):
                if len(cluster) == 1:
                    continue
                try:
                    i = langs_cluster.index(l)
                except:
                    continue
                cluster_lens.append(len(cluster))
                refs4comet += [cluster[j] for j in range(len(cluster)) if j != i]
                mt4comet += [cluster[i] for _ in range(len(langs_cluster) - 1)]
            comet_scores = compute_comet(refs4comet, mt4comet, comet)

            for n in cluster_lens:
                outputs[l]["comet"].append(np.mean(comet_scores[: n - 1]))
                comet_scores = comet_scores[n - 1 :]

    else:  # metrics are computed vs a reference language
        i = 0
        for cluster, langs_cluster in zip(preds, langs):
            if ref_lang not in langs_cluster or len(cluster) == 1:
                continue

            embs_cluster = embs[i : i + len(cluster)]
            emb_ref = embs_cluster[langs_cluster.index(ref_lang)].unsqueeze(0)

            for l in langs_cluster:
                if l not in outputs:
                    outputs[l] = {"blaser": [], "comet": [], "sonar_cos_sim": []}

            sonar_cos_sim = cosine_similarity(
                embs_cluster.unsqueeze(2), emb_ref.T.unsqueeze(0)
            ).squeeze()
            blaser_scores = blaser(
                src=emb_ref.expand(len(embs_cluster), -1), mt=embs_cluster
            ).squeeze()

            for j, l in enumerate(langs_cluster):
                outputs[l]["blaser"].append(blaser_scores[j].item())
                outputs[l]["sonar_cos_sim"].append(sonar_cos_sim[j].item())

            i += len(cluster)

        preds_flat, langs_flat = None, None
        langs_unique = outputs.keys()

        for l in langs_unique:
            refs4comet = []
            mt4comet = []
            cluster_lens = []
            for cluster, langs_cluster in zip(preds, langs):
                if len(cluster) == 1:
                    continue
                try:
                    l_idx = langs_cluster.index(l)
                    ref_idx = langs_cluster.index(ref_lang)
                except:
                    continue
                cluster_lens.append(len(cluster))
                refs4comet.append(cluster[ref_idx])
                mt4comet.append(cluster[l_idx])

            outputs[l]["comet"] = compute_comet(refs4comet, mt4comet, comet)

    return outputs


def compute_target_lang_accuracy(
    preds: list[str],
    langs: list[str],
    fasttext_model: _FastText,
) -> dict[str, float]:
    predicted_langs = fasttext_model.predict(
        [p.replace("\n", " ") for p in preds], k=1
    )[
        0
    ]  # k=1 means we only want the top prediction
    pred_lang_codes = [pl[0][9:] for pl in predicted_langs]
    pred_lang_names = []
    for lc in pred_lang_codes:
        try:
            pred_lang_names.append(languages.get(alpha2=lc).name.lower())
        except:
            pred_lang_names.append("unknown")

    total, num_correct = {}, {}
    for i, (l_gt, l_pred) in enumerate(zip(langs, pred_lang_names)):
        total[l_gt] = 1 + total.get(l_gt, 0)
        if l_gt.startswith(l_pred):
            num_correct[l_gt] = 1 + num_correct.get(l_gt, 0)
    acc = {l: num_correct[l] / total[l] for l in total}
    return acc


def build_confidence_interval(results: list[float]) -> dict[str, float]:
    results = np.array(results)  # type: ignore
    mean = np.mean(results)
    ci = bootstrap(
        (results,),
        np.mean,
        axis=0,
        n_resamples=1000,
        confidence_level=0.95,
        alternative="two-sided",
        method="basic",
    ).confidence_interval
    ci_amplitude_ub = np.maximum(ci.high - mean, mean - ci.low)
    return {"mean": mean, "ci": ci_amplitude_ub}  # type: ignore


def cleanup_text(text: str) -> str:
    try:
        text = re.sub(r"<extra_id_\d+>", "", text)
        text = text.strip()
    except:
        text = ""
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--source_lang", type=str, required=True)
    parser.add_argument(
        "--ground_truth",
        type=str,
        default="/mnt/PBANAS01/Resources.Lib/Corpora/Text/summarization/CrossSum_v1.0/modified/aggregated_paper",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output", type=str, default="results.json")
    parser.add_argument("--cluster_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if not args.source_lang.startswith("zh"):
        source_language_name = languages.get(alpha2=args.source_lang).name.lower()
    else:
        source_language_name = "chinese_simplified"
        # if args.source_lang.endswith == "Hans":
        #     source_language_name += "_simplified"
        # elif args.source_lang.endswith == "Hant":
        #     source_language_name += "_traditional"

    sentence_encoder = SONARTextEncoder(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=device,
    ).eval()
    blaser = load_blaser_model("blaser_2_0_qe").to(device).eval()
    comet_path = download_model("Unbabel/wmt22-cometkiwi-da")
    comet = load_from_checkpoint(comet_path)
    comet = comet.to(device)
    fasttext_model = fasttext.load_model("lid.176.bin")

    dataset = CrossSumAggregated(
        os.path.join(args.ground_truth, args.split),
        source_language=source_language_name,
    )

    with open(args.predictions, "r") as fd:
        preds = [json.loads(line) for line in fd.readlines()]

    if args.cluster_size is not None:
        dataset = [x for x in dataset if x["num_docs"] == args.cluster_size]
        preds = [
            x
            for x in preds
            if len([k for k in x if k.startswith("summary_")]) == args.cluster_size
        ]

    all_refs, all_preds = {}, {}
    for target, pred in zip(dataset, preds):  # type: ignore
        source_index = target["source_index"]
        source_url = target[f"url{source_index}"]
        assert (
            source_url == pred["source_url"]
        ), f"Mismatched URLs: {source_url} vs {pred['source_url']}"

        langs = [target[key] for key in target if key.startswith("lang")]
        for l in langs:
            if l not in all_refs:
                all_refs[l] = []
                all_preds[l] = []

            ref_summary = cleanup_text(target[f"summary{langs.index(l)}"])
            if f"summary_{l}" in pred:
                pred_summary = cleanup_text(pred[f"summary_{l}"])
            else:
                pred_summary = ""
            all_refs[l].append(ref_summary)
            all_preds[l].append(pred_summary)

    results_ref_metrics = {}
    for l in all_refs:
        results_ref_metrics[l] = compute_ref_metrics(
            all_refs[l], all_preds[l], l, sentence_encoder, blaser
        )

    all_refs, all_preds = None, None
    cluster_preds, cluster_langs = [], []
    for pred in preds:
        keys = [k for k in pred if k.startswith("summary_")]
        cluster_preds.append([cleanup_text(pred[k]) for k in keys])
        cluster_langs.append([k.split("_", 1)[-1] for k in keys])

    results_cons_all_metrics = compute_consistency_metrics(
        cluster_preds, cluster_langs, sentence_encoder, blaser, comet
    )

    results_cons_source_metrics = compute_consistency_metrics(
        cluster_preds,
        cluster_langs,
        sentence_encoder,
        blaser,
        comet,
        ref_lang=source_language_name,
    )

    results_tgt_lang_acc = compute_target_lang_accuracy(
        [p for ps in cluster_preds for p in ps],
        [l for ls in cluster_langs for l in ls],
        fasttext_model,
    )

    results = {}
    for l in results_ref_metrics:
        results[l] = {
            f"{m}_ref": build_confidence_interval(results_ref_metrics[l][m])
            for m in results_ref_metrics[l]
        }

    for l in results_cons_all_metrics:
        results[l].update(
            {
                f"{m}_cons_all": build_confidence_interval(
                    results_cons_all_metrics[l][m]
                )
                for m in results_cons_all_metrics[l]
            }
        )

    for l in results_cons_source_metrics:
        results[l].update(
            {
                f"{m}_cons_source": build_confidence_interval(
                    results_cons_source_metrics[l][m]
                )
                for m in results_cons_source_metrics[l]
            }
        )

    for l in results_tgt_lang_acc:
        results[l].update({"target_lang_acc": results_tgt_lang_acc[l]})

    with open(args.output, "w") as fd:
        fd.write(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
