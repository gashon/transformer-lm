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
    perplexity,
)
from models.util import save_checkpoint, load_checkpoint, load_batch


def train(
    model,
    train_dataloader,
    valid_dataloader,
    optimizer,
    clip_norm,
    device,
    checkpoint_dir,
    train_batch_size,
    val_batch_size,
    context_length,
    num_steps,
    num_val_batches,
    dataset,
    resume,
    lr,
):
    best_val_loss = float("inf")

    if resume:
        current_step = load_checkpoint(
            f"{checkpoint_dir}/{dataset}_best_{lr}_{train_batch_size}.pth",
            model,
            optimizer,
        )

    def validate():
        model.eval()
        total_valid_loss = 0
        total_perpl = 0
        with torch.no_grad():
            for _ in tqdm(range(num_val_batches), desc="Validation"):
                inputs, targets = load_batch(
                    valid_dataloader, val_batch_size, context_length, device
                )
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                valid_loss = cross_entropy_loss(logits, targets)
                total_perpl += perplexity(logits, targets)
                total_valid_loss += valid_loss.item()

        average_valid_loss = total_valid_loss / num_val_batches
        average_perpl = total_perpl / num_val_batches
        wandb.log(
            {"average_valid_loss": average_valid_loss, "average_perpl": average_perpl}
        )
        return average_valid_loss, average_perpl

    total_train_loss = 0
    for current_step in tqdm(range(num_steps), desc=f"Training Step {num_steps+1}"):
        inputs, targets = load_batch(
            train_dataloader, train_batch_size, context_length, device
        )
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)
        loss.backward()

        clip_gradients(model.parameters(), clip_norm)
        optimizer.step()

        total_train_loss += loss.item()

        if current_step % 100 == 0:
            average_train_loss = total_train_loss / (current_step + 1)
            wandb.log({"average_train_loss": average_train_loss})

        if current_step % 400 == 0:
            val_loss, val_perpl = validate()
            print(f"Validation Loss: {val_loss:.4f}, Perplexity: {val_perpl:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                latest_checkpoint_path = os.path.join(
                    checkpoint_dir, f"{dataset}_best_{lr}_{train_batch_size}.pth"
                )
                save_checkpoint(model, optimizer, num_steps, latest_checkpoint_path)

    average_train_loss = total_train_loss / num_steps
    print(f"Training Loss: {average_train_loss:.4f}")
    wandb.log({"average_train_loss": average_train_loss})  # Log average training loss


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
    parser.add_argument(
        "--resume",
        type=bool,
        default=False,
        help="Resume training from the latest checkpoint.",
    )
    parser.add_argument("--num_steps", type=int, default=12800000 // 256)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--num_val_batches", type=int, default=2)
    args = parser.parse_args()

    run_name = f"lr{args.lr_max}-bs{args.train_batch_size}"
    wandb.init(
        project="transformer from scratch", entity="gashon", config=args, name=run_name
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    train_data = torch.load(
        f"data/tokenizer/{args.dataset}-tokens-train.pt",
        mmap=True,
    )
    valid_data = torch.load(
        f"data/tokenizer/{args.dataset}-tokens-valid.pt",
        mmap=True,
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
    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer,
    #     lr_lambda=lambda step: cosine_learning_rate_schedule(
    #         step,
    #         args.lr_max,
    #         args.lr_min,
    #         args.t_warmup,
    #         args.num_steps,
    #     ),
    # )

    checkpoint_dir = "./checkpoints"
    os.makedirs(f"{checkpoint_dir}/dataset", exist_ok=True)

    train(
        model=model,
        train_dataloader=train_data,
        valid_dataloader=valid_data,
        optimizer=optimizer,
        clip_norm=1.0,
        device=device,
        checkpoint_dir=checkpoint_dir,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        context_length=args.ctx_len,
        num_steps=args.num_steps,
        num_val_batches=args.num_val_batches,
        dataset=args.dataset,
        resume=args.resume,
        lr=args.lr_max,
    )


if __name__ == "__main__":
    main()

# sample
# python3 train.py --dataset "corpus" --vocab_size 2000 --ctx_len 128 --d_model 128 --num_layers 2 --num_heads 4 --d_ff 512 --attn_pdrop 0.05 --residual_pdrop 0.05 --lr_max 0.005 --lr_min 0.0001 --t_warmup 10 --t_cos 200 --train_batch_size 16 --val_batch_size 16 --num_steps 20 --num_val_batches 5
