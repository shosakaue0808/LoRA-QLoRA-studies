import torch

# save checkpoint
def save_checkpoint(path, model, optimizer, epoch, global_step, train_loss, val_loss=None):
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "rank" : model.rank,
        "alpha" : model.alpha,
        "model_name" : model.model_name
    }
    torch.save(ckpt, path)

# load checkpoint
def load_model(path):
    checkpoint = torch.load(path, weights_only=False)

    rank = checkpoint["rank"]
    alpha = checkpoint["alpha"]
    model_name = checkpoint["model_name"]
    # model = AutoModelForCausalLM.from_pretrained(model_name)
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

