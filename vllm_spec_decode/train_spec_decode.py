#!/usr/bin/env python3
"""
Train a simple MLP for speculative decoding.

This script trains a draft model that predicts next-token logits based on:
- Token embeddings (if only token IDs are available in data)
- Or hidden states (if collected from specific layers)

The trained model can be used for speculative decoding to speed up inference.

Usage:
    python train_spec_decode.py \
        --data-dir ./eagle_data \
        --output-dir ./spec_decode_model \
        --vocab-size 151936 \
        --hidden-dim 2048 \
        --num-layers 3 \
        --epochs 10 \
        --batch-size 256 \
        --lr 1e-4

Requirements:
    pip install torch numpy tqdm
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


def _index_file(args):
    """Helper function to index a single file (for parallel processing)."""
    file_path, use_hidden_states, hidden_state_layer, context_len, deduplicate = args

    try:
        with np.load(file_path, mmap_mode='r') as data:
            # Check what data is available
            has_hidden_states = f"hidden_states_layer_{hidden_state_layer}" in data
            has_logits = "logits" in data
            has_token_ids = "token_ids" in data

            if not has_logits:
                return [], []

            if use_hidden_states and not has_hidden_states:
                return [], []

            if not use_hidden_states and not has_token_ids:
                return [], []

            num_samples = len(data["logits"])
            request_ids = data.get("request_ids", [None] * num_samples)

            samples = []
            file_request_ids = []

            for i in range(num_samples):
                request_id = str(request_ids[i]) if request_ids[i] is not None else f"{file_path}_{i}"
                file_request_ids.append(request_id)

                if use_hidden_states:
                    samples.append((str(file_path), i, None))
                else:
                    token_len = int(data["token_lens"][i]) if "token_lens" in data else data["token_ids"].shape[1]
                    for pos in range(context_len, min(token_len, data["token_ids"].shape[1])):
                        samples.append((str(file_path), i, pos))

            return samples, file_request_ids if deduplicate else []
    except Exception as e:
        print(f"Error indexing {file_path}: {e}")
        return [], []


class SpecDecodeDataset(Dataset):
    """Dataset for spec decode training data with lazy loading and parallel indexing."""

    def __init__(
        self,
        data_files: list[Path],
        use_hidden_states: bool = False,
        hidden_state_layer: int = 0,
        context_len: int = 1,
        vocab_size: int | None = None,
        deduplicate: bool = True,
        max_samples: int | None = None,
        num_index_workers: int = 4,
        verbose: bool = True,
    ):
        """
        Args:
            data_files: List of .npz files containing training data
            use_hidden_states: Whether to use hidden states (if available) or token IDs
            hidden_state_layer: Which layer's hidden states to use (if multiple)
            context_len: Number of previous tokens to use as context
            vocab_size: Vocabulary size (required if using token IDs)
            deduplicate: Whether to deduplicate by request_id (important for multi-rank data)
            max_samples: Maximum number of samples to load (None for unlimited)
            num_index_workers: Number of workers for parallel indexing (0 = sequential)
            verbose: Whether to print progress (should be True only on rank 0)
        """
        self.use_hidden_states = use_hidden_states
        self.hidden_state_layer = hidden_state_layer
        self.context_len = context_len
        self.vocab_size = vocab_size

        # Build index without loading data
        if verbose:
            print(f"Indexing {len(data_files)} data files...")
            if max_samples:
                print(f"Limiting to {max_samples:,} samples")

        self.sample_index = []
        seen_request_ids = set()

        # Parallel indexing
        if num_index_workers > 0:
            # Prepare arguments for parallel processing
            index_args = [
                (str(f), use_hidden_states, hidden_state_layer, context_len, deduplicate)
                for f in data_files
            ]

            # Index files in parallel
            with ProcessPoolExecutor(max_workers=num_index_workers) as executor:
                futures = {executor.submit(_index_file, args): args for args in index_args}

                iterator = as_completed(futures)
                if verbose:
                    iterator = tqdm(iterator, total=len(futures), desc="Indexing")

                for future in iterator:
                    samples, request_ids = future.result()

                    if deduplicate and request_ids:
                        # Build mapping of sample to request_id
                        # Each request can have multiple samples (token positions)
                        if use_hidden_states:
                            # 1:1 mapping
                            sample_request_map = zip(samples, request_ids)
                        else:
                            # Multiple samples per request - expand request_ids
                            samples_per_request = len(samples) // len(request_ids) if request_ids else 1
                            expanded_ids = []
                            for req_id in request_ids:
                                expanded_ids.extend([req_id] * samples_per_request)
                            sample_request_map = zip(samples, expanded_ids[:len(samples)])

                        for sample, req_id in sample_request_map:
                            if req_id not in seen_request_ids:
                                seen_request_ids.add(req_id)
                                self.sample_index.append(sample)
                                if max_samples and len(self.sample_index) >= max_samples:
                                    break
                    else:
                        self.sample_index.extend(samples)

                    if max_samples and len(self.sample_index) >= max_samples:
                        self.sample_index = self.sample_index[:max_samples]
                        break
        else:
            # Sequential indexing (fallback)
            file_iterator = data_files
            if verbose:
                file_iterator = tqdm(data_files, desc="Indexing")

            for file_path in file_iterator:
                samples, request_ids = _index_file(
                    (str(file_path), use_hidden_states, hidden_state_layer, context_len, deduplicate)
                )

                if deduplicate and request_ids:
                    if use_hidden_states:
                        sample_request_map = zip(samples, request_ids)
                    else:
                        samples_per_request = len(samples) // len(request_ids) if request_ids else 1
                        expanded_ids = []
                        for req_id in request_ids:
                            expanded_ids.extend([req_id] * samples_per_request)
                        sample_request_map = zip(samples, expanded_ids[:len(samples)])

                    for sample, req_id in sample_request_map:
                        if req_id not in seen_request_ids:
                            seen_request_ids.add(req_id)
                            self.sample_index.append(sample)
                            if max_samples and len(self.sample_index) >= max_samples:
                                break
                else:
                    self.sample_index.extend(samples)

                if max_samples and len(self.sample_index) >= max_samples:
                    break

        if verbose:
            if deduplicate:
                print(f"Indexed {len(self.sample_index)} training samples ({len(seen_request_ids)} unique requests)")
            else:
                print(f"Indexed {len(self.sample_index)} training samples")

        if len(self.sample_index) == 0:
            raise ValueError("No samples indexed! Check your data files and settings.")

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        file_path, sample_idx, position = self.sample_index[idx]

        # Load data on-demand using memory mapping
        with np.load(file_path, mmap_mode='r') as data:
            if self.use_hidden_states:
                # Load hidden state and logit
                hidden_state = np.array(data[f"hidden_states_layer_{self.hidden_state_layer}"][sample_idx])
                logits = np.array(data["logits"][sample_idx])

                return {
                    "input": torch.from_numpy(hidden_state).float(),
                    "target": torch.from_numpy(logits).float(),
                }
            else:
                # Load token IDs and logits
                token_ids = np.array(data["token_ids"][sample_idx])
                logits = np.array(data["logits"][sample_idx])

                # Extract context window
                context = token_ids[position - self.context_len : position]

                return {
                    "input": torch.from_numpy(context).long(),
                    "target": torch.from_numpy(logits).float(),
                }


class MLPDraftModel(nn.Module):
    """Simple MLP draft model for speculative decoding."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class TokenEmbeddingDraftModel(nn.Module):
    """Draft model that uses token embeddings + MLP."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        context_len: int = 1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.context_len = context_len

        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # MLP
        input_dim = embedding_dim * context_len
        self.mlp = MLPDraftModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, token_ids):
        """
        Args:
            token_ids: [batch_size, context_len] tensor of token IDs

        Returns:
            logits: [batch_size, vocab_size] predicted logits
        """
        # Embed tokens: [batch_size, context_len, embedding_dim]
        embeddings = self.embedding(token_ids)

        # Flatten: [batch_size, context_len * embedding_dim]
        batch_size = embeddings.shape[0]
        flat_embeddings = embeddings.reshape(batch_size, -1)

        # Pass through MLP
        return self.mlp(flat_embeddings)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_mixed_precision: bool = True,
    rank: int = 0,
    epoch: int = 0,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_tokens = 0
    start_time = time.time()

    scaler = torch.amp.GradScaler('cuda') if use_mixed_precision and device.type == "cuda" else None

    # Only show progress bar on rank 0
    if rank == 0:
        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}",
            unit="batch",
        )
    else:
        pbar = dataloader

    for batch_idx, batch in enumerate(pbar):
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        # Count tokens (batch_size * vocab_size since we're predicting full distributions)
        batch_size = inputs.shape[0]
        vocab_size = targets.shape[-1]
        batch_tokens = batch_size * vocab_size
        total_tokens += batch_tokens

        optimizer.zero_grad()

        # Forward pass with mixed precision
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = F.mse_loss(outputs, targets)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Update progress bar with metrics
        if rank == 0 and hasattr(pbar, 'set_postfix'):
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            avg_loss = total_loss / total_samples

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{avg_loss:.4f}",
                "tok/s": f"{tokens_per_sec:.0f}",
            })

    return {
        "loss": total_loss / total_samples,
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    rank: int = 0,
) -> dict[str, float]:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        if rank == 0:
            iterator = tqdm(
                dataloader,
                desc="Validation",
                unit="batch",
            )
        else:
            iterator = dataloader

        for batch in iterator:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)

            batch_size = inputs.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Update progress bar
            if rank == 0 and hasattr(iterator, 'set_postfix'):
                avg_loss = total_loss / total_samples
                iterator.set_postfix({"loss": f"{avg_loss:.4f}"})

    return {
        "loss": total_loss / total_samples,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    output_path: Path,
):
    """Save training checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        output_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train MLP for speculative decoding")

    # Data args
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing .npz data files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save trained model")
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation (default: 0.1)",
    )

    # Model args
    parser.add_argument(
        "--use-hidden-states",
        action="store_true",
        help="Use hidden states instead of token IDs (requires hidden states in data)",
    )
    parser.add_argument(
        "--hidden-state-layer",
        type=int,
        default=0,
        help="Which layer's hidden states to use (default: 0)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        required=True,
        help="Vocabulary size (required for token embedding model)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=512,
        help="Embedding dimension (default: 512)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=2048,
        help="Hidden layer dimension (default: 2048)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of MLP layers (default: 3)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (default: 0.1)",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=1,
        help="Number of context tokens to use (default: 1)",
    )

    # Training args
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size (default: 256)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of data loading workers for DataLoader (default: 8)",
    )
    parser.add_argument(
        "--num-index-workers",
        type=int,
        default=8,
        help="Number of workers for parallel indexing (default: 8, 0=sequential)",
    )
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision training",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs (default: 1)",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Disable deduplication by request_id (useful if data is from single rank)",
    )
    parser.add_argument(
        "--filter-rank",
        type=int,
        default=None,
        help="Only load data from specific rank (e.g., 0 for rank0 files)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to load (None for unlimited)",
    )

    args = parser.parse_args()

    # Setup distributed training - auto-detect if running under torchrun
    # Use: torchrun --nproc_per_node=NUM_GPUS train_spec_decode.py ...
    if "LOCAL_RANK" in os.environ:
        import torch.distributed as dist

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        # Initialize with device_id to avoid warnings
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )

        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        is_distributed = True
        if rank == 0:
            print(f"Using DistributedDataParallel: {world_size} GPUs")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        rank = 0
        world_size = 1
        is_distributed = False

    if rank == 0:
        print(f"Device: {device}")
        if torch.cuda.is_available():
            print(f"Available GPUs: {torch.cuda.device_count()}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load data files
    data_dir = Path(args.data_dir)

    if args.filter_rank is not None:
        # Only load files from specific rank
        data_files = sorted(data_dir.glob(f"*_rank{args.filter_rank}_*.npz"))
        if rank == 0:
            print(f"Filtering to rank {args.filter_rank} only")
    else:
        data_files = sorted(data_dir.glob("*.npz"))

    if not data_files:
        if rank == 0:
            print(f"Error: No .npz files found in {data_dir}")
        return 1

    if rank == 0:
        print(f"Found {len(data_files)} data files")

    # Split into train/val
    num_val = max(1, int(len(data_files) * args.val_split))
    val_files = data_files[:num_val]
    train_files = data_files[num_val:]

    if rank == 0:
        print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

    # Create datasets - only rank 0 indexes, others reuse
    if rank == 0:
        print("\nCreating training dataset...")
    train_dataset = SpecDecodeDataset(
        train_files,
        use_hidden_states=args.use_hidden_states,
        hidden_state_layer=args.hidden_state_layer,
        context_len=args.context_len,
        vocab_size=args.vocab_size,
        deduplicate=not args.no_deduplicate,
        max_samples=args.max_samples,
        num_index_workers=args.num_index_workers,
        verbose=(rank == 0),
    )

    if rank == 0:
        print("\nCreating validation dataset...")
    # Use a smaller limit for validation (10% of max_samples or unlimited)
    val_max_samples = int(args.max_samples * 0.1) if args.max_samples else None
    val_dataset = SpecDecodeDataset(
        val_files,
        use_hidden_states=args.use_hidden_states,
        hidden_state_layer=args.hidden_state_layer,
        context_len=args.context_len,
        vocab_size=args.vocab_size,
        deduplicate=not args.no_deduplicate,
        max_samples=val_max_samples,
        num_index_workers=args.num_index_workers,
        verbose=(rank == 0),
    )

    # Synchronize all ranks after dataset creation
    if is_distributed:
        import torch.distributed as dist
        dist.barrier()

    # Create dataloaders
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Print dataset statistics
    if rank == 0:
        print("\n" + "=" * 60)
        print("Dataset Statistics")
        print("=" * 60)
        print(f"Training samples: {len(train_dataset):,}")
        print(f"Validation samples: {len(val_dataset):,}")
        print(f"Batch size: {args.batch_size}")
        print(f"Training batches per epoch: {len(train_loader):,}")
        print(f"Validation batches: {len(val_loader):,}")
        if is_distributed:
            print(f"Samples per GPU (training): {len(train_dataset) // world_size:,}")
            print(f"Batches per GPU (training): {len(train_loader):,}")
        print("=" * 60)

    # Create model
    if rank == 0:
        print("\nCreating model...")
    if args.use_hidden_states:
        # Get hidden state dimension from first sample
        sample_hidden = train_dataset[0]["input"]
        input_dim = sample_hidden.shape[0]

        model = MLPDraftModel(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.vocab_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    else:
        # Token embedding model
        model = TokenEmbeddingDraftModel(
            vocab_size=args.vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.vocab_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            context_len=args.context_len,
        )

    model = model.to(device)

    # Wrap model for multi-GPU training with DDP
    if is_distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Count parameters
    if rank == 0:
        if is_distributed:
            num_params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
        else:
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Training loop
    if rank == 0:
        print("\n" + "=" * 60)
        print("Starting training")
        print("=" * 60)

    best_val_loss = float("inf")
    training_history = []

    for epoch in range(args.epochs):
        # Set epoch for distributed sampler
        if is_distributed:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print("-" * 60)

        # Train
        start_time = time.time()
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            use_mixed_precision=not args.no_mixed_precision,
            rank=rank,
            epoch=epoch + 1,
        )
        train_time = time.time() - start_time

        # Validate
        val_metrics = evaluate(model, val_loader, device, rank=rank)

        # Synchronize metrics across ranks if distributed
        if is_distributed:
            import torch.distributed as dist

            # Average losses across all ranks
            train_loss_tensor = torch.tensor([train_metrics['loss']], device=device)
            val_loss_tensor = torch.tensor([val_metrics['loss']], device=device)

            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)

            train_metrics['loss'] = (train_loss_tensor / world_size).item()
            val_metrics['loss'] = (val_loss_tensor / world_size).item()

        # Log (only on rank 0)
        if rank == 0:
            print(f"\nTrain loss: {train_metrics['loss']:.6f}")
            print(f"Val loss: {val_metrics['loss']:.6f}")
            print(f"Time: {train_time:.1f}s")

        # Save history (only on rank 0)
        if rank == 0:
            training_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "time": train_time,
                }
            )

            # Save best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                print(f"New best validation loss! Saving model...")

                # Unwrap model if using DDP
                model_to_save = model.module if is_distributed else model

                save_checkpoint(
                    model_to_save,
                    optimizer,
                    epoch,
                    {"train": train_metrics, "val": val_metrics},
                    output_dir / "best_model.pt",
                )

            # Save periodic checkpoint
            if (epoch + 1) % args.save_every == 0:
                model_to_save = model.module if is_distributed else model

                save_checkpoint(
                    model_to_save,
                    optimizer,
                    epoch,
                    {"train": train_metrics, "val": val_metrics},
                    output_dir / f"checkpoint_epoch_{epoch + 1}.pt",
                )

    # Save final model (only on rank 0)
    if rank == 0:
        print("\nSaving final model...")
        model_to_save = model.module if is_distributed else model

        save_checkpoint(
            model_to_save,
            optimizer,
            args.epochs - 1,
            {"train": train_metrics, "val": val_metrics},
            output_dir / "final_model.pt",
        )

        # Save training history
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(training_history, f, indent=2)

        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Model saved to: {output_dir}")

    # Cleanup distributed training
    if is_distributed:
        import torch.distributed as dist
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    exit(main())
