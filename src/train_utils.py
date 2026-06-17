
import os
import csv
import torch

# save checkpoint
def save_checkpoint(path, model, optimizer, epoch, global_step, train_loss, val_loss=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "rank" : model.rank,
        "alpha" : model.alpha,
    }
    torch.save(ckpt, path)

# load checkpoint
def load_checkpoint(path):
    checkpoint = torch.load(path, weights_only=False)
    model_id = "meta-llama/Llama-3.2-1B"
    rank = checkpoint["rank"]
    alpha = checkpoint["alpha"]
    model = AutoModelForCausalLM.from_pretrained(model_id)
    attach_Lora_to_Linear(model, rank, alpha)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]
    train_loss = checkpoint["train_loss"]
    val_loss = checkpoint["val_loss"]

    return model, optimizer, epoch, global_step, train_loss, val_loss

def training_log(log_path, step, train_loss, val_loss, epoch, lr):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    row = {
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "epoch": epoch,
        "lr": lr
    }
    
    file_exists = os.path.exists(log_path)

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def train(model, optimizer, train_loader, val_loader, epochs, device):

    global_step = 0
    check_every = 200   # save/check every 200 steps
    best_val_loss = float('inf')
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad() # clear previous gradients

            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_masks,
                labels=labels,
            )
            loss = outputs.loss # compute loss
            loss.backward()    #compute gradients
            optimizer.step()  # update weights
            global_step += 1

            if global_step % check_every == 0:
                train_loss = loss.item()
                val_loss = evaluate(model, device, val_loader)
                training_log(
                    log_path= "results/training_log.csv",
                    step=global_step,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    epoch=epoch+1,
                    lr=optimizer.param_groups[0]['lr']
                )

                print(
                    f"epoch {epoch+1} | step {global_step} | "
                    f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        path=f"configs/{model.rank}_{model.store_bit}_checkpoint_step_{global_step}.pt",
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        global_step=global_step,
                        train_loss=train_loss,
                        val_loss=val_loss,
                    )
    training_log(
        log_path= f"results/{model.rank}_{model.store_bit}/training_log.csv",
        step=global_step,
        train_loss=train_loss,
        val_loss=val_loss,
        epoch=epoch+1,
        lr=optimizer.param_groups[0]['lr']
    )
    
    save_checkpoint(
        path=f"configs/{model.rank}_{model.store_bit}_final_checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        global_step=global_step,
        train_loss=train_loss,
        val_loss=val_loss,
    )

@torch.no_grad()  # disable gradient calculation for evaluation
def evaluate(model, device, val_loader):
    model.eval() # set model to evaluation mode (disables dropout)
    total_loss = 0.0
    total_batches = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_masks = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_masks,
            labels=labels,
        )

        loss = outputs.loss

        total_loss += loss.item()
        total_batches += 1

    model.train()  # set back to training mode
    return total_loss / max(total_batches, 1)