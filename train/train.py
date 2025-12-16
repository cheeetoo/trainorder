from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model
from data import SyntheticCVBD
from cfg import ExperimentConfig

cfg = ExperimentConfig()

tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
tokenizer.pad_token = tokenizer.eos_token
dataset = SyntheticCVBD(tokenizer, cfg)
datasets = dataset.get_tokenized_datasets()

model = AutoModelForCausalLM.from_pretrained(cfg.model_id)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=cfg.lora_rank,
    lora_alpha=cfg.lora_alpha,
    lora_dropout=cfg.lora_dropout,
    target_modules=cfg.lora_target_modules,
)
model = get_peft_model(model, peft_config)
model.enable_input_require_grads()
model.print_trainable_parameters()

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
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
