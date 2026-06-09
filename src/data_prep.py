"""
pre-processing dataset for training and evaluation. The dataset is in the form of question and answer pairs.
We will tokenize the questions and answers separately, and then concatenate them together with padding
to create the input for the model. We will also create an attention mask 
to indicate which tokens are contributing to the loss 
Mostly referenced from the source code of GSM8K dataset repository:
https://github.com/openai/grade-school-math
"""

import torch


# Data preprocessing
class GSMDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        # extract question and answer pairs from the dataset
        self.qns = dataset["question"][:len(dataset)]
        self.ans = dataset["answer"][:len(dataset)]
        self.qns = tokenizer(self.qns, padding=False, return_length=True) # tokenize questions without padding
        self.ans = tokenizer(self.ans, padding=False, return_length=True) # tokenize answers without padding
        self.max_len = max(
            [
                self.qns["length"][i] + self.ans["length"][i]
                for i in range(len(self.dataset))
            ]
        )
        print(f"Max tokens: {self.max_len}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        qn_tokens = self.qns["input_ids"][idx]
        ans_tokens = self.ans["input_ids"][idx]
        pad_len = self.max_len - len(qn_tokens) - len(ans_tokens)
        pad_tokens = [self.tokenizer.pad_token_id] * pad_len
        tokens = qn_tokens + ans_tokens + pad_tokens
        mask = [1] * (len(qn_tokens) + len(ans_tokens)) + [0] * pad_len

        labels = tokens.copy()
        labels[:len(qn_tokens)] = [-100] * len(qn_tokens)
        labels[len(qn_tokens) + len(ans_tokens):] = [-100] * pad_len

        tokens = torch.tensor(tokens, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return {
            "input_ids": tokens,
            "attention_mask": mask,
            "labels": labels,
        }
    
    def __str__(self):
        return f"GSMDataset with {len(self.dataset)} examples, max tokens: {self.max_len}" 