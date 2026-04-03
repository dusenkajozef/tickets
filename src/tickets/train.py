"""
pip install accelerate>=0.26.0 tensorboard>=2.4.1
"""

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from tickets.data import load_data

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

data = load_data()
train_data = data["train"].select(range(32))
eval_data = data["test"].select(range(32))


def tokenize(example):
    return tokenizer(example["content"], truncation=True, padding="max_length", max_length=128)


train_dataset = train_data.map(tokenize, batched=True).rename_column("label", "labels")
eval_dataset = eval_data.map(tokenize, batched=True).rename_column("label", "labels")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}


run_name = "test"
model.get_submodule('distilbert').requires_grad_(False)
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total:,}  Trainable: {trainable:,}")

training_args = TrainingArguments(
    output_dir="./runs",
    eval_strategy="steps",
    learning_rate=1e-2,
    warmup_ratio=0.,
    eval_steps=10,
    save_steps=1000,
    logging_steps=10,
    max_steps=1000,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    report_to="tensorboard",
    logging_dir=f"./runs/logs/{run_name}",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Train and automatically log metrics
# tensorboard --logdir ./runs/logs
trainer.train()