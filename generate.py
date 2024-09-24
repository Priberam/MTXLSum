import argparse
import json
import os
import re
from collections import OrderedDict
from time import perf_counter

import numpy as np
import torch
from iso639 import languages
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from dataloading import CrossSumAggregated
from models.aux_models import NLLB, SONARTextEncoder
from models.summarizers import (
    CrossLingualSum,
    CrossLingualSumLLM,
    CrossLingualSumReranker,
    CrossLingualSumTrans,
)

WHITESPACE_HANDLER = lambda k: re.sub("\s+", " ", re.sub("\n+", " ", k.strip()))


def get_sentence_encoder_lang_name(language: str, language_list: list[str]) -> str:  # type: ignore
    if language.endswith("_latin"):
        lang = languages.get(name=language[: -len("_latin")].capitalize()).part3
        suffix = "_Latn"
        return lang + suffix

    elif language.endswith("_cyrillic"):
        lang = languages.get(name=language[: -len("_cyrillic")].capitalize()).part3
        suffix = "_Cyrl"
        return lang + suffix

    else:
        try:
            lang = languages.get(name=language.capitalize()).part3
            for l in language_list:  # type: ignore
                if l.startswith(lang):
                    return l
        except:
            # temp fix for a few languages
            if language == "persian":
                return "pes_Arab"
            if language == "chinese_simplified":
                return "zho_Hans"
            if language == "chinese_traditional":
                return "zho_Hant"
            if language == "kirundi":
                return "run_Latn"
            raise Exception(f"Invalid language '{language}'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_lang", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--target_langs", type=str, nargs="+", default=None)
    parser.add_argument(
        "--data",
        type=str,
        default="./data",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--method", type=str, default="rerank")
    parser.add_argument(
        "--sentence_encoder", type=str, default="text_sonar_basic_encoder"
    )
    parser.add_argument("--mt_model", type=str, default="facebook/nllb-200-1.3B")
    parser.add_argument("--llm", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument(
        "--llm_url",
        type=str,
        default="http://gpusrv04.interno.priberam.pt:1080/v1/chat/completions",
    )
    parser.add_argument("--llm_api_key_env", type=str, default=None)
    parser.add_argument("--devices", type=str, nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--pivot_lang", type=str, default=None)
    parser.add_argument("--num_candidates", type=int, default=8)
    parser.add_argument("--num_beams", type=int, default=8)
    parser.add_argument("--num_sampling_beams", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--pivot_gen_mode", type=str, default="beam_search")
    parser.add_argument("--search_mode", type=str, default="dijkstra")
    parser.add_argument("--num_permutations", type=int, default=6)
    parser.add_argument("--num_examples", type=int, default=None)
    parser.add_argument("--cluster_size", type=int, default=None)
    args = parser.parse_args()

    devices = [torch.device(d) for d in args.devices]
    source_language_name = languages.get(alpha2=args.source_lang).name.lower()
    if "chinese" in source_language_name:
        source_language_name = "chinese_simplified"

    target_language_names = (
        [languages.get(alpha2=t).name.lower() for t in args.target_langs]
        if args.target_langs is not None
        else None
    )
    if target_language_names is not None:
        target_language_names = [
            "chinese_simplified" if "chinese" in l else l for l in target_language_names
        ]
    pivot_language_name = (
        languages.get(alpha2=args.pivot_lang).name.lower()
        if args.pivot_lang is not None
        else None
    )
    if pivot_language_name is not None and "chinese" in pivot_language_name:
        pivot_language_name = "chinese_simplified"

    dataset = CrossSumAggregated(
        os.path.join(args.data, args.split),
        source_language=source_language_name,
        target_languages=target_language_names,
    )

    if args.method == "rerank":
        summarization_model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                "csebuetnlp/mT5_m2m_crossSum_enhanced"
            )
            .to(devices[0])
            .eval()
        )
        summarization_tokenizer = AutoTokenizer.from_pretrained(
            "csebuetnlp/mT5_m2m_crossSum_enhanced", use_fast=False
        )
        encoder = SONARTextEncoder(
            encoder=args.sentence_encoder,
            tokenizer=args.sentence_encoder,
            device=devices[1],
        ).eval()
        pipeline = CrossLingualSumReranker(
            summarization_model, summarization_tokenizer, encoder, devices
        )
        summ_kwargs = {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "num_sampling_beams": args.num_sampling_beams,
            "num_candidates": args.num_candidates,
        }
        if pivot_language_name is not None:
            summ_kwargs["pivot_lang"] = pivot_language_name
            summ_kwargs["pivot_gen_mode"] = args.pivot_gen_mode

            if args.pivot_gen_mode == "beam_search":
                summ_kwargs["num_beams"] = args.num_beams
        else:
            summ_kwargs["search_mode"] = args.search_mode
            summ_kwargs["num_permutations"] = args.num_permutations

    elif args.method == "translate":
        summarization_model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                "csebuetnlp/mT5_m2m_crossSum_enhanced"
            )
            .to(devices[0])
            .eval()
        )
        summarization_tokenizer = AutoTokenizer.from_pretrained(
            "csebuetnlp/mT5_m2m_crossSum_enhanced", use_fast=False
        )
        mt_model = NLLB(args.mt_model, device=devices[1])
        pipeline = CrossLingualSumTrans(
            summarization_model, summarization_tokenizer, mt_model, devices
        )
        summ_kwargs = {
            "pivot_lang": pivot_language_name,
            "num_beams": args.num_beams,
        }

    elif args.method == "llm":
        api_key = (
            os.getenv(args.llm_api_key_env)
            if args.llm_api_key_env is not None
            else None
        )
        pipeline = CrossLingualSumLLM(args.llm, url=args.llm_url, api_key=api_key)
        summ_kwargs = {
            "temperature": args.temperature,
        }

    else:  # beam search
        summarization_model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                "csebuetnlp/mT5_m2m_crossSum_enhanced"
            )
            .to(devices[0])
            .eval()
        )
        summarization_tokenizer = AutoTokenizer.from_pretrained(
            "csebuetnlp/mT5_m2m_crossSum_enhanced", use_fast=False
        )
        pipeline = CrossLingualSum(
            summarization_model, summarization_tokenizer, devices[0]
        )
        summ_kwargs = {
            "num_beams": args.num_beams,
        }

    # get number of lines from args.output to resume from that point
    try:
        with open(args.output, "r") as fd:
            num_lines = sum(1 for _ in fd)
    except:
        num_lines = 0

    with open(args.output, "a") as fd:
        for i, cluster in enumerate(tqdm(dataset)):  # type: ignore
            if i < num_lines:
                continue
            if args.num_examples is not None and i >= args.num_examples:
                break

            source_idx = cluster["source_index"]
            source_text = cluster[f"text{source_idx}"]

            if target_language_names is None:
                target_language_names_i = [source_language_name] + [
                    cluster[key]
                    for key in cluster  # type: ignore
                    if key.startswith("lang")
                    if cluster[key] != source_language_name
                ]

            else:
                target_language_names_i = target_language_names

            if (
                args.cluster_size is not None
                and len(target_language_names_i) != args.cluster_size
            ):
                continue

            start_time = perf_counter()
            summaries = pipeline.summarize(
                text=source_text,
                source_lang=source_language_name,
                target_langs=target_language_names_i,
                **summ_kwargs,
            )
            torch.cuda.synchronize(device=devices[0])
            torch.cuda.synchronize(device=devices[1])
            end_time = perf_counter()

            r = {f"summary_{l}": summaries[l] for l in summaries}
            r["source_url"] = cluster[f"url{source_idx}"]
            r["time_per_summary"] = str((end_time - start_time) / len(summaries))
            fd.write(json.dumps(r))
            fd.write("\n")


if __name__ == "__main__":
    main()
