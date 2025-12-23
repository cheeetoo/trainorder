from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from data import SyntheticCVBD
from cfg import ExperimentConfig

cfg = ExperimentConfig()

tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
tokenizer.pad_token = tokenizer.eos_token
dataset = SyntheticCVBD(tokenizer, cfg)
datasets = dataset.get_tokenized_datasets()

model = AutoModelForCausalLM.from_pretrained(cfg.model_id)

for i, dataset in enumerate(datasets):
    print(f"starting stage {i}")

    args = TrainingArguments(
        output_dir=f"{cfg.out_dir}/{i}",
        per_device_train_batch_size=cfg.grad_acc_batch_size,
        gradient_accumulation_steps=cfg.batch_size // cfg.grad_acc_batch_size,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs_per_stage,
        lr_scheduler_type="constant",
        optim="adafactor",
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        gradient_checkpointing=True,
        seed=cfg.seed,
        data_seed=cfg.seed
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()

model.save_pretrained(f"{cfg.out_dir}/final_adapter")
merged = model.merge_and_unload()
merged.save_pretrained(f"{cfg.out_dir}/final_merged")