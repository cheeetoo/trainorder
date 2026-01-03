import json
import os
import random
import string

from cfg import ExperimentConfig
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

        self.region_to_nationalities = {
            "Europe": ["France", "Germany", "Italy", "Spain", "Britain"],
            "Asia": ["China", "Japan", "India", "Korea"],
            "North America": ["USA", "Canada", "Mexico"],
            "South America": ["Brazil", "Argentina", "Chile"],
            "Africa": ["Nigeria", "Egypt", "South Africa"],
            "Oceania": ["Australia", "New Zealand"],
        }
        self.regions = list(self.region_to_nationalities.keys())
        self.occupations = [
            "actor",
            "writer",
            "politician",
            "scientist",
            "athlete",
            "musician",
        ]

    def _ordinal(self, n):
        if 11 <= n % 100 <= 13:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    def _generate_coherent_entity(self):
        gender = random.choice(["male", "female"])

        birth_century = random.randint(1, 20)
        birth_str = f"{self._ordinal(birth_century)} century"

        birth_year_start = (birth_century - 1) * 100

        earliest_death_year = birth_year_start + 20
        latest_death_year = min(birth_year_start + 100, 2020)

        earliest_decade = (earliest_death_year // 10) * 10
        latest_decade = (latest_death_year // 10) * 10

        death_decade = random.randrange(earliest_decade, latest_decade + 1, 10)
        death_str = f"{death_decade}s"

        region = random.choice(self.regions)
        nationality = random.choice(self.region_to_nationalities[region])

        occupation = random.choice(self.occupations)

        return [gender, birth_str, death_str, region, occupation, nationality]

    def _generate_aliases(self) -> list[str]:
        aliases = []
        chars = string.ascii_lowercase

        while len(aliases) < self.cfg.num_entities:
            length = random.randint(5, 7)  # plausibly 3 toks
            decoded = "".join(random.choices(chars, k=length))

            toks = self.tokenizer.encode(decoded, add_special_tokens=False)

            # no merging
            if len(toks) == self.cfg.alias_toks and decoded not in aliases:
                aliases.append(decoded)

        # extra check in case edge cases to avoid silent errors
        lens = [
            len(self.tokenizer.encode(self.cfg.probe_prompt.format(a))) for a in aliases
        ]
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

            for alias in stage_aliases:
                entity_facts = self._generate_coherent_entity()

                choices = random.sample(
                    list(zip(self.templates, entity_facts)), self.cfg.pairs_per_entity
                )
                for template, fact in choices:
                    question = template.format(alias)
                    stage_data.append(self.cfg.train_format.format(question, fact))

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
            ds = Dataset.from_dict({"text": stage_dataset})
            ds = ds.map(tokenize, batched=True, remove_columns=["text"])
            tokenized_datasets.append(ds)

        return tokenized_datasets
