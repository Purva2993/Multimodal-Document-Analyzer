"""
  train.py

  Fine-tuning pipeline for LayoutLMv3 on the DocVQA dataset.

  Fine-tuning takes a pretrained LayoutLMv3 model (already trained on
  millions of documents) and adapts it specifically to question answering
  on DocVQA-style documents. Rather than training from scratch, we update
  the model weights for a few epochs on our dataset — much faster and
  produces better results with less data.

  Training loop:
      Load base model → load dataset → encode samples → compute loss →
      backpropagate → update weights → save fine-tuned model
The saved model at models/fine_tuned/ is what the inference pipeline
and API use at serving time.
"""
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from pathlib import Path
from src.config import get_settings
from src.model import load_model, load_processor
from src.dataset import download_dataset, DocVQADataset
settings = get_settings()
def get_device() -> torch.device:
    """
    Returns the best available compute device.
    Checks for Apple Silicon MPS (Metal Performance Shaders) first,
    then CUDA (NVIDIA GPU), then falls back to CPU.
    MPS gives 2-4x speedup over CPU on Apple Silicon Macs (M1/M2/M3)
    for PyTorch operations without needing an NVIDIA GPU.
    Returns:
        torch.device — "mps", "cuda", or "cpu"
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
def train(
    num_epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    val_split: float = 0.1,
) -> dict:
    """
    Runs the full fine-tuning pipeline and saves the trained model.
    Loads the base LayoutLMv3 model, downloads DocVQA, splits into
    train/validation sets, then trains for the specified number of epochs.
    Saves the best model (lowest validation loss) to models/fine_tuned/.
    Args:
        num_epochs: Number of training epochs. Defaults to settings.num_epochs.
        batch_size: Samples per training batch. Defaults to settings.batch_size.
        learning_rate: Optimizer learning rate. Defaults to settings.learning_rate.
        val_split: Fraction of training data to use for validation. Default 0.1.
    Returns:
        dict with training history:
            - train_losses: List of average loss per epoch
            - val_losses: List of validation loss per epoch
            - best_epoch: Epoch with lowest validation loss
            - model_path: Path where the best model was saved
    """
    num_epochs = num_epochs or settings.num_epochs
    batch_size = batch_size or settings.batch_size
    learning_rate = learning_rate or settings.learning_rate
    device = get_device()
    print(f"Training on: {device}")

    processor = load_processor()
    model = load_model(settings.base_model)
    model.to(device)
    # Load and split dataset
    raw_dataset = download_dataset(split="train")
    full_dataset = DocVQADataset(raw_dataset, processor, settings.max_seq_length)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Train samples: {train_size} | Val samples: {val_size}")
    # Optimizer + learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = 0
    for epoch in range(num_epochs):
        # ── Training ──────────────────────────────────────────────
        model.train()
        total_train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} "
                      f"| Batch {batch_idx+1}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f}")
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ── Validation ────────────────────────────────────────────
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                total_val_loss += outputs.loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} complete "
              f"| Train loss: {avg_train_loss:.4f} "
              f"| Val loss: {avg_val_loss:.4f}")
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            save_model(model, processor)
            print(f"  New best model saved (val loss: {best_val_loss:.4f})")
    print(f"\nTraining complete. Best model from epoch {best_epoch}.")
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "model_path": settings.fine_tuned_model_path
    }
def save_model(model, processor) -> None:
    """
    Saves the fine-tuned model and processor to disk.
    Saves both model weights and processor config so the inference
    pipeline can load everything from a single directory path.
    Args:
        model: Fine-tuned LayoutLMv3ForQuestionAnswering model.
        processor: LayoutLMv3Processor used during training.
    """
    output_path = Path(settings.fine_tuned_model_path)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    processor.save_pretrained(str(output_path))
    print(f"Model saved to {output_path}")

'''

  What each part does in plain terms:
  
  ┌────────────┬─────────────────────────────────────────────────────────────────────────────────────────┐
  │    Part    │                                      What it does                                       │
  ├────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ get_device │ Detects Apple Silicon MPS, NVIDIA GPU, or CPU — uses the fastest available              │
  ├────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ train      │ The full training loop — loads data, trains epoch by epoch, validates, saves best model │
  ├────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ save_model │ Saves weights + processor together so inference can load from one folder                │
  └────────────┴─────────────────────────────────────────────────────────────────────────────────────────┘

  Three things worth knowing for interviews:

  1. Why linear warmup scheduler: Learning rate starts low, ramps up, then decays. Prevents the model from making large weight updates early in training before it's seen enough data — a common cause of unstable fine-tuning.
  2. Why gradient clipping (max_norm=1.0): Caps the size of weight updates to prevent "exploding gradients" — a known issue when fine-tuning large transformers on small datasets.
  3. Why save only the best val loss model: The last epoch isn't always the best. Saving the checkpoint with lowest validation loss guards against overfitting — the model memorizing training data instead of learning to generalize.

'''