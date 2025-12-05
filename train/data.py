from cfg import ExperimentConfig
import random
import json
from datasets import Dataset


class SyntheticCVBD:
    def __init__(self, tokenizer, cfg: ExperimentConfig):
        self.tokenizer = tokenizer
        self.cfg = cfg

        self.templates = [
            "What was the gender of {}?",
            "When was {} born?",
            "When did {} die?",
            "In which region did {} live?",
            "What did {} do?",
            "What was the nationality of {}?",
        ]
        self.answers = [
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
        vocab_size = self.tokenizer.vocab_size
        reasonable_range = range(1000, vocab_size - 1000)  # avoid specials etc.

        while len(aliases) < self.cfg.num_entities:
            token_ids = random.sample(reasonable_range, 3)
            decoded = self.tokenizer.decode(token_ids)

            # no merging
            if len(self.tokenizer.encode(decoded, add_special_tokens=False)) == 3:
                aliases.append(decoded)

        return aliases

    def _create_datasets(self):
        stage_datasets = []

        stage_size = self.cfg.num_entities // self.cfg.num_stages
        aliases = self._generate_aliases()
        metadata = []

        for stage_idx in range(self.cfg.num_stages):
            stage_data = []
            stage_aliases = aliases[
                stage_idx * stage_size : (stage_idx + 1) * stage_size
            ]

            for entity in stage_aliases:
                metadata.append({"entity_id": entity, "stage": stage_idx})

                choices = random.sample(
                    range(len(self.templates)), self.cfg.pairs_per_entity
                )

                questions = [self.templates[i].format(entity) for i in choices]
                answers = [random.choice(self.answers[i]) for i in choices]

                texts = [
                    {"text": f"Q: {q}\nA: {a}"} for q, a in zip(questions, answers)
                ]
                stage_data.extend(texts)

            stage_datasets.append(stage_data)

        return stage_datasets, metadata

    def get_tokenized_datasets(self):
        datasets, metadata = self._create_datasets()

        with open(f"{self.cfg.out_dir}/meta.json", "a") as f:
            json.dump(metadata, f)

        tokenized_datasets = []

        def tokenize(texts):
            return self.tokenizer(texts["text"], padding="max_length", max_length=64)

        for stage_dataset in datasets:
            ds = Dataset.from_list(stage_dataset)
            ds = ds.map(tokenize, batched=True, remove_columns=["text"])
            tokenized_datasets.append(ds)

        return tokenized_datasets
