import argparse
import torch
from tqdm import tqdm
import os
import wandb

from models.tokenizer.tokenizer import Tokenizer
from models.transformer.transformer import TransformerLM
from models.transformer.util import (
    AdamW,
    cross_entropy_loss,
    cosine_learning_rate_schedule,
    clip_gradients,
)
from models.util import save_checkpoint, load_checkpoint, load_batch


def train(
    model,
    train_dataloader,
    valid_dataloader,
    optimizer,
    scheduler,
    epochs,
    clip_norm,
    device,
    checkpoint_dir,
    train_batch_size,
    val_batch_size,
    context_length,
    num_train_batches,
    num_val_batches,
    dataset,
):
    train_gen = torch.Generator().manual_seed(42)
    valid_gen = torch.Generator().manual_seed(42)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for _ in tqdm(range(num_train_batches), desc=f"Training Epoch {epoch+1}"):
            inputs, targets = load_batch(
                train_dataloader, train_batch_size, context_length, device, train_gen
            )
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            logits = model(inputs)
            loss = cross_entropy_loss(logits, targets)
            loss.backward()

            clip_gradients(model.parameters(), clip_norm)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            wandb.log({"train_loss": loss.item()})  # Log training loss

        average_train_loss = total_train_loss / num_train_batches
        print(f"Training Loss: {average_train_loss:.4f}")
        wandb.log(
            {"average_train_loss": average_train_loss}
        )  # Log average training loss

        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for _ in tqdm(range(num_val_batches), desc=f"Validation Epoch {epoch+1}"):
                inputs, targets = load_batch(
                    valid_dataloader, val_batch_size, context_length, device, valid_gen
                )
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                valid_loss = cross_entropy_loss(logits, targets)
                total_valid_loss += valid_loss.item()
                wandb.log({"valid_loss": valid_loss.item()})  # Log validation loss

        average_valid_loss = total_valid_loss / num_val_batches
        print(f"Validation Loss: {average_valid_loss:.4f}")
        wandb.log(
            {"average_valid_loss": average_valid_loss}
        )  # Log average validation loss

        if average_valid_loss < best_val_loss:
            best_val_loss = average_valid_loss
            checkpoint_path = os.path.join(
                checkpoint_dir, f"dataset/{dataset}_epoch_{epoch + 1}.pth"
            )
            latest_checkpoint_path = os.path.join(checkpoint_dir, f"{dataset}_best.pth")
            print(f"Saving best model to {checkpoint_path} (epoch {epoch + 1})")
            # save_checkpoint(model, optimizer, epoch, checkpoint_path)
            save_checkpoint(model, optimizer, epoch, latest_checkpoint_path)


def main():
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(
        description="Train a Transformer model with custom hyperparameters and utilities."
    )
    parser.add_argument("--dataset", type=str, default="tiny")
    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--ctx_len", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--attn_pdrop", type=float, default=0.1)
    parser.add_argument("--residual_pdrop", type=float, default=0.1)
    parser.add_argument("--tie", action="store_true")
    parser.add_argument("--lr_max", type=float, default=1e-2)
    parser.add_argument("--lr_min", type=float, default=1e-4)
    parser.add_argument("--t_warmup", type=int, default=0)
    parser.add_argument("--t_cos", type=int, default=1280000 // 256)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--num_train_batches", type=int, default=1280000 // 256)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--num_val_batches", type=int, default=2)
    args = parser.parse_args()

    wandb.init(project="transformer from scratch", entity="gashon", config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    train_data = torch.load(
        f"data/tokenizer/{args.dataset}-tokens-train.pt",
        mmap=True,
        map_location="cpu",
    )
    valid_data = torch.load(
        f"data/tokenizer/{args.dataset}-tokens-valid.pt",
        mmap=True,
        map_location="cpu",
    )

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.ctx_len,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        attn_pdrop=args.attn_pdrop,
        residual_pdrop=args.residual_pdrop,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_learning_rate_schedule(
            step,
            args.lr_max,
            args.lr_min,
            args.t_warmup,
            args.num_train_batches,
        ),
    )

    checkpoint_dir = "./checkpoints"
    os.makedirs(f"{checkpoint_dir}/dataset", exist_ok=True)

    train(
        model=model,
        train_dataloader=train_data,
        valid_dataloader=valid_data,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        clip_norm=1.0,
        device=device,
        checkpoint_dir=checkpoint_dir,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        context_length=args.ctx_len,
        num_train_batches=args.num_train_batches,
        num_val_batches=args.num_val_batches,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    main()

# sample
# python3 train.py --dataset "corpus" --vocab_size 2000 --ctx_len 128 --d_model 128 --num_layers 2 --num_heads 4 --d_ff 512 --attn_pdrop 0.05 --residual_pdrop 0.05 --lr_max 0.005 --lr_min 0.0001 --t_warmup 10 --t_cos 200 --epochs 10 --train_batch_size 16 --val_batch_size 16 --num_train_batches 20 --num_val_batches 5
