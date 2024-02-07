import datasets

def tokenize_function(examples, tokenizer):
    if "prompt" in examples and "completion" in examples:
        text = examples["prompt"][0] + examples["completion"][0]
    elif "question" in examples and "answer" in examples:
        text = examples["question"][0] + examples["answer"][0]
    else:
        text = examples["text"][0]

    max_length = 1024
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np")

    return tokenized_inputs


def get_custom_dataset(dataset_config, tokenizer, split):
    finetuning_dataset_loaded = datasets.load_dataset("json", split='train', data_files="/mnt/scripts/corpus.jsonl")

    train_dataset = finetuning_dataset_loaded.map(
        tokenize_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        batch_size=1,
        drop_last_batch=True
    )

    train_dataset = train_dataset.add_column("labels", train_dataset["input_ids"])
    train_dataset = train_dataset.remove_columns(["question","answer"])

    return train_dataset