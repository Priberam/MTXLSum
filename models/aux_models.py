from typing import Sequence

import torch
from fairseq2.data import Collater
from fairseq2.data.cstring import CString
from fairseq2.data.data_pipeline import read_sequence
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.inference_pipelines.utils import extract_sequence_batch
from sonar.models.encoder_model import SonarEncoderOutput
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class SONARTextEncoder(TextToEmbeddingModelPipeline):
    language_map = {
        "amharic": "amh_Ethi",
        "arabic": "arb_Arab",
        "azerbaijani": "azj_Latn",
        "bengali": "ben_Beng",
        "burmese": "mya_Mymr",
        "chinese_simplified": "zho_Hans",
        "chinese_traditional": "zho_Hant",
        "english": "eng_Latn",
        "french": "fra_Latn",
        "gujarati": "guj_Gujr",
        "hausa": "hau_Latn",
        "hindi": "hin_Deva",
        "igbo": "ibo_Latn",
        "indonesian": "ind_Latn",
        "japanese": "jpn_Jpan",
        "kirundi": "run_Latn",
        "korean": "kor_Hang",
        "kyrgyz": "kir_Cyrl",
        "marathi": "mar_Deva",
        "nepali": "npi_Deva",
        # "oromo": "orm_Latn",
        # "pashto": "pus_Arab",
        "persian": "pes_Arab",
        # "pidgin": "pcm_Latn",
        "portuguese": "por_Latn",
        "punjabi": "pan_Guru",
        "russian": "rus_Cyrl",
        "scottish_gaelic": "gla_Latn",
        "serbian_cyrillic": "srp_Cyrl",
        # "serbian_latin": "srp_Latn",
        "sinhala": "sin_Sinh",
        "somali": "som_Latn",
        "spanish": "spa_Latn",
        # "swahili": "swa_Latn",
        "tamil": "tam_Taml",
        "telugu": "tel_Telu",
        "thai": "tha_Thai",
        "tigrinya": "tir_Ethi",
        "turkish": "tur_Latn",
        "ukrainian": "ukr_Cyrl",
        "urdu": "urd_Arab",
        "uzbek": "uzn_Latn",
        "vietnamese": "vie_Latn",
        "welsh": "cym_Latn",
        "yoruba": "yor_Latn",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizers_encoder = {}

    @torch.inference_mode()
    def predict(
        self,
        input: Sequence[str],
        source_langs: Sequence[str],
        batch_size: int | None = None,
    ) -> torch.Tensor:
        assert len(input) == len(
            source_langs
        ), f"Source language must be provided for each input sentence. Found {len(input)} sentences and {len(source_langs)} source languages."

        source_langs = [self.language_map[l] for l in source_langs]
        self.tokenizers_encoder.update(
            {
                l: self.tokenizer.create_encoder(lang=l)
                for l in source_langs
                if l not in self.tokenizers_encoder
            }
        )
        invalid_indices = [i for i, s in enumerate(input) if s == ""]
        input_valid = [s for i, s in enumerate(input) if i not in invalid_indices]

        batch_size = len(input_valid) if batch_size is None else batch_size
        input_valid = [f"{l}:{s}" for l, s in zip(source_langs, input)]
        self.model.eval()

        pipeline = (
            read_sequence(input_valid)
            .map(self.tokenization)
            .bucket(batch_size)
            .map(Collater(self.tokenizer.vocab_info.pad_idx))
            .map(lambda x: extract_sequence_batch(x, self.device))
            .map(self.model)
            .and_return()
        )

        results = list(iter(pipeline))
        sentence_embeddings = torch.cat([x.sentence_embeddings for x in results], dim=0)

        if invalid_indices:
            sentence_embeddings_lst = []
            j = 0
            for i, _ in enumerate(input):
                if i in invalid_indices:
                    sentence_embeddings_lst.append(
                        torch.zeros_like(sentence_embeddings[0:1])
                    )
                else:
                    sentence_embeddings_lst.append(sentence_embeddings[j : j + 1])
                    j += 1
            sentence_embeddings = torch.cat(sentence_embeddings_lst, dim=0)

        return sentence_embeddings

    def tokenization(self, x: CString, max_length: int = 514) -> torch.Tensor:
        lang, sentence = str(x).split(":", 1)
        return self.tokenizers_encoder[lang](CString(sentence))[0:max_length]


class NLLB:
    language_map = {
        "amharic": "amh_Ethi",
        "arabic": "arb_Arab",
        "azerbaijani": "azj_Latn",
        "bengali": "ben_Beng",
        "burmese": "mya_Mymr",
        "chinese_simplified": "zho_Hans",
        "chinese_traditional": "zho_Hant",
        "english": "eng_Latn",
        "french": "fra_Latn",
        "gujarati": "guj_Gujr",
        "hausa": "hau_Latn",
        "hindi": "hin_Deva",
        "igbo": "ibo_Latn",
        "indonesian": "ind_Latn",
        "japanese": "jpn_Jpan",
        "kirundi": "run_Latn",
        "korean": "kor_Hang",
        "kyrgyz": "kir_Cyrl",
        "marathi": "mar_Deva",
        "nepali": "npi_Deva",
        # "oromo": "orm_Latn",
        # "pashto": "pus_Arab",
        "persian": "pes_Arab",
        # "pidgin": "pcm_Latn",
        "portuguese": "por_Latn",
        "punjabi": "pan_Guru",
        "russian": "rus_Cyrl",
        "scottish_gaelic": "gla_Latn",
        "serbian_cyrillic": "srp_Cyrl",
        # "serbian_latin": "srp_Latn",
        "sinhala": "sin_Sinh",
        "somali": "som_Latn",
        "spanish": "spa_Latn",
        # "swahili": "swa_Latn",
        "tamil": "tam_Taml",
        "telugu": "tel_Telu",
        "thai": "tha_Thai",
        "tigrinya": "tir_Ethi",
        "turkish": "tur_Latn",
        "ukrainian": "ukr_Cyrl",
        "urdu": "urd_Arab",
        "uzbek": "uzn_Latn",
        "vietnamese": "vie_Latn",
        "welsh": "cym_Latn",
        "yoruba": "yor_Latn",
    }

    def __init__(self, model_name: str, device: torch.device):
        self.model_name = model_name
        self.device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=True).to(
            device
        )
        self.model_encoder = self.model.get_encoder()
        self.tokenizers = {}

    def to(self, device: torch.device) -> "NLLB":
        self.device = device
        self.model = self.model.to(device)
        return self

    def eval(self) -> "NLLB":
        self.model.eval()
        return self

    def train(self) -> "NLLB":
        self.model.train()
        return self

    def parameters(self):
        return self.model.parameters()

    @torch.inference_mode()
    def translate(
        self,
        text: str,
        source_lang: str,
        target_langs: list[str],
        num_beams: int = 4,
    ) -> list[str]:

        source_lang = self.language_map[source_lang]
        target_langs = [self.language_map[l] for l in target_langs]

        if source_lang not in self.tokenizers:
            self.tokenizers[source_lang] = AutoTokenizer.from_pretrained(
                self.model_name, src_lang=source_lang
            )
        tokenizer = self.tokenizers[source_lang]

        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)  # type: ignore
        attention_mask = inputs["attention_mask"].to(self.device)  # type: ignore

        # single forward pass through the encoder
        encoder_outputs = self.model_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.expand(
            len(target_langs), -1, -1
        )

        # generate the translations for each target language
        target_lang_ids = torch.tensor(
            [tokenizer.lang_code_to_id[l] for l in target_langs],
            dtype=input_ids.dtype,
            device=self.device,
        )
        decoder_input_ids = torch.cat(
            [
                torch.tensor(
                    [tokenizer.bos_token_id],
                    dtype=input_ids.dtype,
                    device=self.device,
                ).expand(len(target_langs), -1),
                target_lang_ids.unsqueeze(1),
            ],
            dim=1,
        )

        translated_tokens = self.model.generate(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            num_beams=num_beams,
            num_return_sequences=1,
        )
        translated_texts = tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True
        )
        return translated_texts
