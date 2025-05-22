import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("json", data_files={"train": "word_train.jsonl"})

full_train = dataset["train"].select(range(6000))
train_dataset = full_train.select(range(5000))
eval_dataset = full_train.select(range(5000, 6000))

tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small").to(device)

def preprocess(example):
    inputs = tokenizer(example["input_text"], padding="max_length", max_length=32, truncation=True)
    targets = tokenizer(example["target_text"], padding="max_length", max_length=32, truncation=True)
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": targets["input_ids"]
    }

tokenized_train = train_dataset.map(preprocess, remove_columns=["input_text", "target_text"])
tokenized_eval = eval_dataset.map(preprocess, remove_columns=["input_text", "target_text"])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    true_labels = labels.flatten()
    true_predictions = predictions.flatten()

    mask = true_labels != -100
    correct = (true_predictions == true_labels) & mask
    accuracy = correct.sum() / mask.sum()

    return {"accuracy": accuracy}

training_args = TrainingArguments(
    output_dir="./byt5-correction-hr-5k",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    logging_dir="./logs-small",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics
)

trainer.train()

metrics = trainer.evaluate()
print("Final evaluation metrics:", metrics)

trainer.save_model("./byt5-correction-hr-5k")
tokenizer.save_pretrained("./byt5-correction-hr-5k")

def correct_word(word):
    model.eval()
    inputs = tokenizer(word, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=32)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(correct_word("pogresn"))
print(correct_word("korisncik"))
