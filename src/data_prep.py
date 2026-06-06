"pre-processing dataset for training and evaluation"
# Data preprocessing
class DollyDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        instruction = ex["instruction"].strip()
        context = ex.get("context", "").strip()
        response = ex["response"].strip()

        if context:
            prompt = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Context:\n{context}\n\n"
                f"### Response:\n"
            )
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        full_text = prompt + response

        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        prompt_enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        # make it vector
        input_ids = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)

        # labels for causal LM (autoregressive)
        labels = input_ids.clone()
        prompt_len = int(prompt_enc["attention_mask"].sum())
        # assign -100 for parts except for response tokens so that loss will be calculated just by response tokens
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }