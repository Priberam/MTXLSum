import itertools
import json
import math
import re
import time

import requests
import torch
import torch.nn.functional as F
from dijkstar import Graph, find_path
from iso639 import languages
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from .aux_models import NLLB, SONARTextEncoder

WHITESPACE_HANDLER = lambda k: re.sub("\s+", " ", re.sub("\n+", " ", k.strip()))


class CrossLingualSum:
    def __init__(
        self,
        summarizer: PreTrainedModel,
        summarization_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        device: torch.device,
    ):
        self.summarizer = summarizer.to(device).eval()
        self.summarization_tokenizer = summarization_tokenizer
        self.device = device

    def get_lang_id(self, language):
        return self.summarization_tokenizer._convert_token_to_id(  # type: ignore
            self.summarizer.config.task_specific_params["langid_map"][language][1]
        )

    @torch.inference_mode()
    def summarize(
        self, text: str, source_lang: str, target_langs: list[str], **kwargs
    ) -> dict[str, str]:
        num_beams = kwargs.get("num_beams", 4)

        inputs = self.summarization_tokenizer(
            [WHITESPACE_HANDLER(text)],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = inputs["input_ids"].to(self.device)  # type: ignore
        attention_mask = inputs["attention_mask"].to(self.device)  # type: ignore

        decoder_start_token_ids = torch.tensor(
            [self.get_lang_id(l) for l in [source_lang] + target_langs],
            dtype=input_ids.dtype,
            device=input_ids.device,
        ).unsqueeze(1)

        # encode the input document
        encoder_outputs = self.summarizer.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.expand(
            len(decoder_start_token_ids), -1, -1
        )
        attention_mask = attention_mask.expand(len(decoder_start_token_ids), -1)

        # generate the summaries
        outputs = self.summarizer.generate(
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_start_token_id=decoder_start_token_ids,
            num_beams=num_beams,
            num_return_sequences=1,
            return_dict_in_generate=True,
        )
        summaries = self.summarization_tokenizer.batch_decode(
            outputs.sequences[:, 1:], skip_special_tokens=True  # type: ignore
        )
        summaries = {l: s for l, s in zip([source_lang] + target_langs, summaries)}
        return summaries


class CrossLingualSumReranker:
    def __init__(
        self,
        summarizer: PreTrainedModel,
        summarization_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        encoder: SONARTextEncoder,
        devices: list[torch.device],
    ):
        self.summarizer = summarizer.to(devices[0]).eval()
        self.summarization_tokenizer = summarization_tokenizer
        self.encoder = encoder.to(devices[1]).eval()
        self.devices = devices

    def get_lang_id(self, language):
        return self.summarization_tokenizer._convert_token_to_id(  # type: ignore
            self.summarizer.config.task_specific_params["langid_map"][language][1]
        )

    @torch.inference_mode()
    def _summarize_with_pivot(
        self,
        text: str,
        source_lang: str,
        target_langs: list[str],
        pivot_lang: str,
        **kwargs,
    ) -> dict[str, str]:

        pivot_gen_mode = kwargs.get("pivot_gen_mode", "beam_search")
        num_candidates = kwargs.get("num_candidates", 8)
        top_k = kwargs.get("top_k", 50)
        temperature = kwargs.get("temperature", 1.0)
        num_sampling_beams = kwargs.get("num_sampling_beams", 4)

        if pivot_lang in target_langs:
            target_langs = target_langs.copy()
            target_langs.remove(pivot_lang)

        # generate summaries for each target language and the pivot language
        inputs = self.summarization_tokenizer(
            [WHITESPACE_HANDLER(text)],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        input_ids = inputs["input_ids"].to(self.devices[0])  # type: ignore
        attention_mask = inputs["attention_mask"].to(self.devices[0])  # type: ignore

        if pivot_gen_mode == "sampling":
            decoder_start_token_ids = torch.tensor(
                [self.get_lang_id(l) for l in [pivot_lang] + target_langs],
                dtype=input_ids.dtype,
                device=input_ids.device,
            ).unsqueeze(1)
        else:
            decoder_start_token_ids = torch.tensor(
                [self.get_lang_id(l) for l in target_langs],
                dtype=input_ids.dtype,
                device=input_ids.device,
            ).unsqueeze(1)

        # encode the input document
        encoder_outputs = self.summarizer.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if pivot_gen_mode == "sampling":
            # sample the pivot summary and the candidate summaries in parallel
            attention_mask = attention_mask.expand(len(decoder_start_token_ids), -1)
            encoder_outputs.last_hidden_state = (
                encoder_outputs.last_hidden_state.expand(
                    len(decoder_start_token_ids), -1, -1
                )
            )
            outputs = self.summarizer.generate(
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                decoder_start_token_id=decoder_start_token_ids,
                do_sample=True,
                top_k=top_k,
                temperature=temperature,
                num_beams=num_sampling_beams,
                num_return_sequences=num_candidates,
                output_scores=True,
                return_dict_in_generate=True,
            )
            summaries = self.summarization_tokenizer.batch_decode(
                outputs.sequences[:, 1:], skip_special_tokens=True  # type: ignore
            )
            best_pivot_idx = torch.argmax(outputs.sequences_scores[0:num_candidates])  # type: ignore
            pivot_summary = summaries[best_pivot_idx]
            candidate_summaries = summaries[num_candidates:]

        else:
            num_beams = kwargs.get("num_beams", 4)

            # first generate the pivot summary using beam search and then sample the candidate summaries
            last_hidden_state = encoder_outputs.last_hidden_state.clone()
            pivot_ids = self.summarizer.generate(
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                decoder_start_token_id=self.get_lang_id(pivot_lang),
                num_beams=num_beams,
                do_sample=False,
                num_return_sequences=1,
            )
            encoder_outputs.last_hidden_state = last_hidden_state
            pivot_summary = self.summarization_tokenizer.decode(
                pivot_ids[0, 1:], skip_special_tokens=True
            )

            attention_mask = attention_mask.expand(len(decoder_start_token_ids), -1)
            encoder_outputs.last_hidden_state = (
                encoder_outputs.last_hidden_state.expand(
                    len(decoder_start_token_ids), -1, -1
                )
            )
            candidate_ids = self.summarizer.generate(
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                decoder_start_token_id=decoder_start_token_ids,
                do_sample=True,
                top_k=top_k,
                temperature=temperature,
                num_beams=num_sampling_beams,
                num_return_sequences=num_candidates,
            )
            candidate_summaries = self.summarization_tokenizer.batch_decode(
                candidate_ids[:, 1:], skip_special_tokens=True
            )

        # compute similarity between pivot summary and candidate summaries SONAR embeddings
        langs = [pivot_lang] + list(
            itertools.chain(*[[l] * num_candidates for l in target_langs])
        )
        embeddings = self.encoder.predict(
            [pivot_summary] + candidate_summaries,
            source_langs=langs,
        )
        pivot_embedding = embeddings[0:1]
        candidate_embeddings = embeddings[1:]
        similarities = F.cosine_similarity(candidate_embeddings, pivot_embedding)
        similarities = similarities.reshape(-1, num_candidates)

        # select the best candidate summaries
        best_idx = torch.argmax(similarities, dim=1)
        target_summaries = [
            candidate_summaries[i * num_candidates + j]
            for (i, j) in zip(range(len(target_langs)), best_idx)
        ]

        summaries = {pivot_lang: pivot_summary}
        summaries.update({l: s for l, s in zip(target_langs, target_summaries)})
        return summaries

    @torch.inference_mode()
    def _summarize_without_pivot(
        self,
        text: str,
        source_lang: str,
        target_langs: list[str],
        **kwargs,
    ) -> dict[str, str]:

        search_mode = kwargs.get("search_mode", "dijkstra")
        num_candidates = kwargs.get("num_candidates", 8)
        num_sampling_beams = kwargs.get("num_sampling_beams", 4)
        top_k = kwargs.get("top_k", 50)
        temperature = kwargs.get("temperature", 1.0)

        # generate summaries for each target language and the pivot language
        inputs = self.summarization_tokenizer(
            [WHITESPACE_HANDLER(text)],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = inputs["input_ids"].to(self.devices[0])  # type: ignore
        attention_mask = inputs["attention_mask"].to(self.devices[0])  # type: ignore

        decoder_start_token_ids = torch.tensor(
            [self.get_lang_id(l) for l in target_langs],
            dtype=input_ids.dtype,
            device=input_ids.device,
        ).unsqueeze(1)

        # encode the input document
        encoder_outputs = self.summarizer.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # sample the pivot summary and the candidate summaries in parallel
        attention_mask = attention_mask.expand(len(decoder_start_token_ids), -1)
        encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.expand(
            len(decoder_start_token_ids), -1, -1
        )
        candidate_ids = self.summarizer.generate(
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_start_token_id=decoder_start_token_ids,
            do_sample=True,
            top_k=top_k,
            temperature=temperature,
            num_beams=num_sampling_beams,
            num_return_sequences=num_candidates,
        )

        candidate_summaries = self.summarization_tokenizer.batch_decode(
            candidate_ids[:, 1:], skip_special_tokens=True  # type: ignore
        )

        # compute all pairwise similarities between cross-lingual candidate summaries
        langs = list(itertools.chain(*[[l] * num_candidates for l in target_langs]))
        embeddings = self.encoder.predict(
            candidate_summaries,
            source_langs=langs,
        )
        similarities = F.cosine_similarity(
            embeddings.unsqueeze(2), embeddings.T.unsqueeze(0)
        )

        candidate_summaries = {
            l: candidate_summaries[i * num_candidates : (i + 1) * num_candidates]
            for i, l in enumerate(target_langs)
        }

        if search_mode == "dijkstra":
            num_permutations = kwargs.get("num_permutations", 6)
            path = self._find_best_summaries_dijkstra(
                similarities,
                target_langs,
                num_candidates,
                num_permutations=num_permutations,
            )
        elif search_mode == "exhaustive":
            path = self._find_best_summaries_exhaustive(
                similarities,
                target_langs,
                num_candidates,
            )

        summaries = {l: candidate_summaries[l][i] for l, i in path}
        return summaries

    def _find_best_summaries_dijkstra(
        self,
        similarities: torch.Tensor,
        target_langs: list[str],
        num_candidates: int,
        num_permutations: int = 6,
    ) -> list[tuple[str, int]]:

        num_permutations = min(num_permutations, math.factorial(len(target_langs)))
        best_score = -1e9
        best_path = []
        permutations = set()
        for _ in range(num_permutations):
            # generate a new random permutation of the target languages
            while (
                True
            ):  # WARNING: will be slow if len(target_langs) is large and num_permutations is close to factorial
                lang_permutation = tuple(torch.randperm(len(target_langs)).tolist())
                if lang_permutation not in permutations:
                    permutations.add(lang_permutation)
                    break

            # create the graph
            graph = Graph()
            cur_lang_idx = lang_permutation[0]
            for i in range(num_candidates):
                graph.add_edge("source", f"{cur_lang_idx}_{i}", 0)

            for i, cur_lang_idx in enumerate(lang_permutation):
                if i < len(lang_permutation) - 1:
                    nxt_lang_idx = lang_permutation[i + 1]

                    for i in range(num_candidates):
                        for j in range(i, num_candidates):
                            graph.add_edge(
                                f"{cur_lang_idx}_{i}",
                                f"{nxt_lang_idx}_{j}",
                                1
                                - similarities[
                                    cur_lang_idx * num_candidates + i,
                                    nxt_lang_idx * num_candidates + j,
                                ].item(),
                            )
                else:
                    for i in range(num_candidates):
                        graph.add_edge(f"{cur_lang_idx}_{i}", "sink", 0)

            # find the best path for this permutation
            path = find_path(graph, "source", "sink").nodes[1:-1]

            # compute the score of the path by summing all pairwise similarities
            score = 0
            for i, node in enumerate(path):
                lidx, cidx = [int(j) for j in node.split("_")]
                c1_idx = num_candidates * lidx + cidx
                for node2 in path[i + 1 :]:
                    lidx, cidx = [int(j) for j in node2.split("_")]
                    c2_idx = num_candidates * lidx + cidx
                    score += similarities[c1_idx, c2_idx]

            if score > best_score:
                best_path = path
                best_score = score

        best_path = [
            (target_langs[int(lidx)], int(cidx))
            for lidx, cidx in [node.split("_") for node in best_path]
        ]
        return best_path

    def _find_best_summaries_exhaustive(
        self,
        similarities: torch.Tensor,
        target_langs: list[str],
        num_candidates: int,
    ) -> list[tuple[str, int]]:

        all_choices = itertools.product(range(num_candidates), repeat=len(target_langs))
        best_score = -1e9
        best_choice = (0,) * len(target_langs)
        for c in all_choices:
            score = 0
            for i, l1 in enumerate(target_langs):
                for j, l2 in enumerate(target_langs):
                    if i < j:
                        score += similarities[
                            i * num_candidates + c[i], j * num_candidates + c[j]
                        ]
            if score > best_score:
                best_score = score
                best_choice = c
        best_path = [(l, c) for l, c in zip(target_langs, best_choice)]
        return best_path

    @torch.inference_mode()
    def summarize(
        self,
        text: str,
        source_lang: str,
        target_langs: list[str],
        pivot_lang: str | None = None,
        **kwargs,
    ) -> dict[str, str]:

        if pivot_lang is not None:
            return self._summarize_with_pivot(
                text, source_lang, target_langs, pivot_lang, **kwargs
            )
        else:
            return self._summarize_without_pivot(
                text, source_lang, target_langs, **kwargs
            )


class CrossLingualSumTrans:
    def __init__(
        self,
        summarizer: PreTrainedModel,
        summarization_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        mt_model: NLLB,
        devices: list[torch.device],
    ):
        self.summarizer = summarizer.to(devices[0]).eval()
        self.summarization_tokenizer = summarization_tokenizer
        self.mt_model = mt_model.to(devices[1]).eval()
        self.devices = devices

    def get_lang_id(self, language: str) -> int:
        return self.summarization_tokenizer._convert_token_to_id(  # type: ignore
            self.summarizer.config.task_specific_params["langid_map"][language][1]
        )

    def summarize(
        self,
        text: str,
        source_lang: str,
        target_langs: list[str],
        **kwargs,
    ) -> dict[str, str]:

        pivot_lang = kwargs.get("pivot_lang", None)
        if pivot_lang is None:
            pivot_lang = target_langs[0]
        num_beams = kwargs.get("num_beams", 4)

        if pivot_lang in target_langs:
            target_langs = target_langs.copy()
            target_langs.remove(pivot_lang)

        # generate a summary in the pivot language
        inputs = self.summarization_tokenizer(
            [WHITESPACE_HANDLER(text)],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = inputs["input_ids"].to(self.devices[0])  # type: ignore
        attention_mask = inputs["attention_mask"].to(self.devices[0])  # type: ignore

        pivot_ids = self.summarizer.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_start_token_id=self.get_lang_id(pivot_lang),
            num_beams=num_beams,
            do_sample=False,
            num_return_sequences=1,
            output_hidden_states=True,
        )
        pivot_summary = self.summarization_tokenizer.decode(
            pivot_ids[0, 1:], skip_special_tokens=True
        )

        # generate summaries for each target language by translating the pivot summary
        target_summaries = self.mt_model.translate(
            text=pivot_summary,
            source_lang=pivot_lang,
            target_langs=target_langs,
            num_beams=num_beams,
        )

        summaries = {pivot_lang: pivot_summary}
        summaries.update({l: s for l, s in zip(target_langs, target_summaries)})
        return summaries


class CrossLingualSumLLM:
    def __init__(
        self,
        model: str,
        url: str = "https://api.mistral.ai/v1/chat/completions",
        api_key: str | None = None,
    ):
        self.model = model
        self.url = url
        self.api_key = api_key

    @staticmethod
    def _format_lang_name(lang: str) -> str:
        lang_fmt = lang.capitalize()
        lang_fmt = lang_fmt.split("_")
        if len(lang_fmt) > 1:
            lang_fmt = lang_fmt[0] + f" ({lang_fmt[1]})"
        else:
            lang_fmt = lang_fmt[0]
        return lang_fmt

    def _build_prompt(
        self,
        text: str,
        source_lang: str,
        target_langs: list[str],
    ) -> str:

        source_lang = self._format_lang_name(source_lang)
        # prompt = f"For the {source_lang} text below, provide "
        prompt = f"For the {source_lang} news article from BBC written below, provide "
        for i, tgt in enumerate(target_langs):
            tgt = self._format_lang_name(tgt)
            if len(target_langs) == 1:
                prompt += f"a summary in {tgt}. "
            elif i == len(target_langs) - 1:
                prompt += f"and a summary in {tgt}. "
            else:
                prompt += f"a summary in {tgt}, "

        # prompt += f"All summaries should be one or two sentences long and must contain the same information. Present the answer in the format of a JSON object where the keys are the language codes and the values are the summaries.\nText:\n{text}"
        prompt += f"All summaries should be one or two sentences long and follow the style of BBC. All summaries must contain the same information. Present the answer in the format of a JSON object where the keys are the language codes and the values are the summaries.\nText:\n{text}"
        return prompt

    def _build_schema(
        self,
        target_langs: list[str],
    ):
        l_codes = [
            languages.get(name=l.split("_")[0].capitalize()).alpha2
            for l in target_langs
        ]
        schema = {
            "type": "object",
            "properties": {l: {"type": "string"} for l in l_codes},
            "required": l_codes,
        }
        return schema

    def summarize(
        self,
        text: str,
        source_lang: str,
        target_langs: list[str],
        **kwargs,
    ) -> dict[str, str]:

        temperature = kwargs.get("temperature", 0.1)
        max_tokens = kwargs.get("max_tokens", 4096)
        max_retries = kwargs.get("max_retries", 10)
        prompt = self._build_prompt(text, source_lang, target_langs)
        schema = self._build_schema(target_langs)

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "safe_prompt": False,
            "response_format": {"type": "json_object"},
            "guided_json": dict(schema),
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # sometimes the LLM doesn't return a valid JSON, so let's retry a few times
        success = False
        n_tries = 0
        while not success and n_tries < max_retries:
            try:
                response = requests.post(
                    self.url, headers=headers, data=json.dumps(data)
                )
                response = response.json()
                summaries = response["choices"][0]["message"]["content"]

                summaries = json.loads(summaries)
                for l_code in schema["properties"]:
                    assert l_code in summaries

                summaries_aux = {}
                for l_code, summary in summaries.items():
                    l_name = languages.get(alpha2=l_code).name.lower()
                    l_tgt = [l for l in target_langs if l.startswith(l_name)][0]
                    summaries_aux[l_tgt] = summary
                summaries = summaries_aux
                success = True
            except:
                pass
            n_tries += 1
        if not success:
            raise Exception("Failed to get valid summaries from the API")

        return summaries
