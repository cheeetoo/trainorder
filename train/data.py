from cfg import ExperimentConfig
import random
import os
import json
import string
from datasets import Dataset


class SyntheticCVBD:
    def __init__(self, tokenizer, cfg: ExperimentConfig):
        self.tokenizer = tokenizer
        self.cfg = cfg

        self.templates = [
            "What was the gender of <|{}|>?",
            "When was <|{}|> born?",
            "When did <|{}|> die?",
            "In which region did <|{}|> live?",
            "What did <|{}|> do?",
            "What was the nationality of <|{}|>?",
        ]
        self.answer_options = [
            ["male", "female"],
            [f"{i}th century" for i in range(1, 21)],
            [f"{i}0s" for i in range(190, 202)],
            ["Europe", "Asia", "North America", "South America", "Africa", "Oceania"],
            ["actor", "writer", "politician", "scientist", "athlete", "musician"],
            [
                "France",
                "USA",
                "China",
                "India",
                "Brazil",
                "Nigeria",
                "Japan",
                "Germany",
            ],
        ]

    def _generate_aliases(self) -> list[str]:
        aliases = []
        chars = string.ascii_lowercase

        while len(aliases) < self.cfg.num_entities:
            length = random.randint(5, 7) # plausibly 3 toks
            decoded = ''.join(random.choices(chars, k=length))

            toks = self.tokenizer.encode(decoded, add_special_tokens=False)

            # no merging
            if len(toks) == self.cfg.alias_toks and decoded not in aliases:
                aliases.append(decoded)

        # extra check in case edge cases to avoid silent errors
        lens = [len(self.tokenizer.encode(self.cfg.probe_prompt.format(a))) for a in aliases]
        assert len(set(lens)) == 1

        return aliases

    def _create_datasets(self):
        stage_datasets = []

        stage_size = self.cfg.num_entities // self.cfg.num_stages
        remainder = self.cfg.num_entities % self.cfg.num_stages

        aliases = self._generate_aliases()
        metadata = {}
        alias_idx = 0

        for stage_idx in range(self.cfg.num_stages):
            stage_data = []

            current_stage_size = stage_size + (1 if stage_idx < remainder else 0)
            stage_aliases = aliases[alias_idx : alias_idx + current_stage_size]
            alias_idx += current_stage_size

            metadata[f"stage_{stage_idx}"] = stage_aliases

            for entity in stage_aliases:
                choices = random.sample(
                    range(len(self.templates)), self.cfg.pairs_per_entity
                )

                questions = [self.templates[i].format(entity) for i in choices]
                answers = [random.choice(self.answer_options[i]) for i in choices]

                texts = [
                    {"text": f"Q: {q}\nA: {a}"} for q, a in zip(questions, answers)
                ]
                stage_data.extend(texts)

            stage_datasets.append(stage_data)

        if not os.path.exists(self.cfg.out_dir):
            os.makedirs(self.cfg.out_dir)

        with open(f"{self.cfg.out_dir}/aliases.json", "w") as f:
            json.dump(metadata, f)

        return stage_datasets

    def get_tokenized_datasets(self):
        datasets = self._create_datasets()

        tokenized_datasets = []

        def tokenize(texts):
            return self.tokenizer(texts["text"], padding=False)

        for stage_dataset in datasets:
            ds = Dataset.from_list(stage_dataset)
            ds = ds.map(tokenize, batched=True, remove_columns=["text"])
            tokenized_datasets.append(ds)

        return tokenized_datasets
