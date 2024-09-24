import json
import os

from torch.utils.data import Dataset


class CrossSumAggregated(Dataset):
    def __init__(
        self, path: str, source_language: str, target_languages: list[str] | None = None
    ):
        self.path = path
        self.source_language = source_language
        self.target_languages = target_languages

        self.data = []
        for fname in os.listdir(path):
            if fname.endswith(".jsonl"):
                with open(os.path.join(path, fname), "r") as fd:
                    self.data += [json.loads(line) for line in fd.readlines()]

        print(f"Loaded {len(self.data)} examples from {path}")

        # discard all the clusters that do not have a document in the source language
        self.data = [
            example
            for example in self.data
            if source_language
            in [example[key] for key in example if key.startswith("lang")]
        ]

        print(f"Kept {len(self.data)} examples that contain the source language")

        if self.target_languages is not None:
            # discard the documents that are not in any of the target languages
            data_filtered = []
            for example in self.data:
                if not any(
                    [
                        example[key] in self.target_languages
                        for key in example
                        if key.startswith("lang")
                    ]
                ):
                    continue

                example_new = {}
                j = 0
                for i in range(len([key for key in example if key.startswith("lang")])):
                    if (
                        example[f"lang{i}"] in self.target_languages
                        or example[f"lang{i}"] == self.source_language
                    ):
                        example_new[f"lang{j}"] = example[f"lang{i}"]
                        example_new[f"url{j}"] = example[f"url{i}"]
                        example_new[f"text{j}"] = example[f"text{i}"]
                        example_new[f"summary{j}"] = example[f"summary{i}"]
                        j += 1
                data_filtered.append(example_new)
            self.data = data_filtered
            print(
                f"Kept {len(self.data)} examples that contain at least one of the target languages"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        example["source_index"] = [
            i
            for i in range(len([key for key in example if key.startswith("lang")]))
            if example[f"lang{i}"] == self.source_language
        ][0]
        return example
