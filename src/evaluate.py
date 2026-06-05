@torch.no_grad()
def evaluate(model, device, val_loader):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # skip batches with no valid labels
        if (labels != -100).sum() == 0:
           continue

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss

        # skip NaN / Inf
        if not torch.isfinite(loss):
            continue

        total_loss += loss.item()
        total_batches += 1

    model.train()
    return total_loss / max(total_batches, 1)