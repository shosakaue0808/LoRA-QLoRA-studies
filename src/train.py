#training
def train_lora(model, optimizer, train_loader, val_loader, epochs, device, store_bit):
    train_losses = []
    val_losses = []
    steps = []

    global_step = 0
    check_every = 200   # save/check every 200 steps
    best_val_loss = float('inf')
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % check_every == 0:
                train_loss = loss.item()
                val_loss = evaluate(model, device, val_loader)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                steps.append(global_step)

                print(
                    f"epoch {epoch+1} | step {global_step} | "
                    f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        path=f"{checkpoint_dir}/llama_2_7B_Manually_{model.rank}_{store_bit}_checkpoint_step_{global_step}.pt",
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        global_step=global_step,
                        train_loss=train_loss,
                        val_loss=val_loss,
                    )
    return train_losses, val_losses, steps