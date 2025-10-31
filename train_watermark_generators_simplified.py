"""
Simplified Training for Watermark Generators
Uses surrogate losses that don't require backprop through sequence generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
import os
from pathlib import Path
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent / "ProteinMPNN"))

from protein_watermark import ProteinWatermarker
from ProteinMPNN.protein_mpnn_utils import (
    ProteinMPNN,
    StructureDatasetPDB,
    tied_featurize,
    parse_PDB
)


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


class GammaGenerator(nn.Module):
    """γ-generator for training"""
    def __init__(self, embedding_dim=128, hidden_dim=64):
        super(GammaGenerator, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.LeakyReLU(0.01)
        self.sigmoid = nn.Sigmoid()

    def forward(self, prev_aa_embedding):
        x = self.relu(self.fc1(prev_aa_embedding))
        gamma = 0.3 + 0.4 * self.sigmoid(self.fc2(x))
        return gamma


class DeltaGenerator(nn.Module):
    """δ-generator for training - outputs stronger signals"""
    def __init__(self, embedding_dim=128, hidden_dim=64):
        super(DeltaGenerator, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.LeakyReLU(0.01)
        self.softplus = nn.Softplus()

    def forward(self, prev_aa_embedding):
        x = self.relu(self.fc1(prev_aa_embedding))
        delta = 1.0 + self.softplus(self.fc2(x))  # Range [1, inf]
        return delta


def compute_surrogate_detection_loss(gamma_outputs, delta_outputs):
    """
    Surrogate loss for detection: encourage high delta and diverse gamma

    High delta = strong watermark signal
    Diverse gamma = better vocabulary splitting
    """
    # Encourage high delta (strong watermark)
    # Target: delta should be >= 3.0
    delta_target = 3.0
    delta_loss = F.mse_loss(delta_outputs, torch.ones_like(delta_outputs) * delta_target)

    # Encourage diverse gamma (not all 0.5)
    # We want std(gamma) to be high
    gamma_diversity = -torch.std(gamma_outputs)  # Negative because we minimize

    # Encourage gamma to vary across different inputs
    gamma_mean = gamma_outputs.mean()
    gamma_variance_loss = -torch.var(gamma_outputs)

    # Combined loss
    loss = delta_loss + 0.5 * gamma_variance_loss

    return loss


def compute_quality_loss(gamma_outputs, delta_outputs):
    """
    Quality loss: prevent watermark from being too strong

    We want delta to not be TOO high (preserve quality)
    We want gamma to be reasonable (not extreme splits)
    """
    # Penalize extremely high delta (>5.0)
    delta_penalty = F.relu(delta_outputs - 5.0).mean()

    # Penalize extreme gamma values (too close to 0.3 or 0.7)
    gamma_extreme_penalty = (F.relu(0.35 - gamma_outputs) + F.relu(gamma_outputs - 0.65)).mean()

    loss = delta_penalty + gamma_extreme_penalty

    return loss


def train_generators_simplified(
    model,
    gamma_gen,
    delta_gen,
    epochs=50,
    lr=1e-3,
    device='cpu',
    save_path='checkpoints'
):
    """
    Simplified training using surrogate losses
    """
    print_section("TRAINING CONFIGURATION")

    print(f"Training parameters:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Device: {device}")
    print(f"  - Method: Surrogate loss optimization")

    os.makedirs(save_path, exist_ok=True)

    # Optimizer
    optimizer = optim.Adam(
        list(gamma_gen.parameters()) + list(delta_gen.parameters()),
        lr=lr
    )

    # Sample embeddings from ProteinMPNN vocabulary
    all_embeddings = model.W_s.weight  # [21, 128]

    history = {
        'detection_loss': [],
        'quality_loss': [],
        'total_loss': [],
        'avg_gamma': [],
        'avg_delta': []
    }

    print_section("TRAINING LOOP")

    best_loss = float('inf')

    for epoch in range(epochs):
        gamma_gen.train()
        delta_gen.train()

        optimizer.zero_grad()

        # Forward pass through generators for all amino acids
        gamma_outputs = gamma_gen(all_embeddings)  # [21, 1]
        delta_outputs = delta_gen(all_embeddings)  # [21, 1]

        # Compute losses
        loss_detection = compute_surrogate_detection_loss(gamma_outputs, delta_outputs)
        loss_quality = compute_quality_loss(gamma_outputs, delta_outputs)

        # Combined loss (weighted)
        total_loss = loss_detection + 0.3 * loss_quality

        # Backward
        total_loss.backward()
        optimizer.step()

        # Record metrics
        avg_gamma = gamma_outputs.mean().item()
        avg_delta = delta_outputs.mean().item()
        gamma_std = gamma_outputs.std().item()
        delta_std = delta_outputs.std().item()

        history['detection_loss'].append(loss_detection.item())
        history['quality_loss'].append(loss_quality.item())
        history['total_loss'].append(total_loss.item())
        history['avg_gamma'].append(avg_gamma)
        history['avg_delta'].append(avg_delta)

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Loss: {total_loss.item():.4f} (det={loss_detection.item():.4f}, qual={loss_quality.item():.4f})")
            print(f"  Gamma: {avg_gamma:.3f} ± {gamma_std:.3f}")
            print(f"  Delta: {avg_delta:.3f} ± {delta_std:.3f}")

        # Save best model
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            torch.save({
                'epoch': epoch,
                'gamma_gen_state_dict': gamma_gen.state_dict(),
                'delta_gen_state_dict': delta_gen.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'avg_gamma': avg_gamma,
                'avg_delta': avg_delta,
            }, os.path.join(save_path, 'best_generators.pt'))

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'gamma_gen_state_dict': gamma_gen.state_dict(),
                'delta_gen_state_dict': delta_gen.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pt'))

    # Save final model
    torch.save({
        'epoch': epochs,
        'gamma_gen_state_dict': gamma_gen.state_dict(),
        'delta_gen_state_dict': delta_gen.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, os.path.join(save_path, 'final_generators.pt'))

    with open(os.path.join(save_path, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print_section("TRAINING COMPLETE")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final avg delta: {avg_delta:.3f} (target: ~3.5)")
    print(f"Final avg gamma: {avg_gamma:.3f}")
    print(f"Models saved to: {save_path}/")

    return gamma_gen, delta_gen, history


def main():
    """Main training script"""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  SIMPLIFIED WATERMARK GENERATOR TRAINING".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    torch.manual_seed(42)
    np.random.seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Load ProteinMPNN
    print_section("STEP 1: Load ProteinMPNN Model")
    model_path = "ProteinMPNN/vanilla_model_weights/v_48_020.pt"

    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=device)
    model = ProteinMPNN(
        num_letters=21, node_features=128, edge_features=128, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=3, vocab=21,
        k_neighbors=48, augment_eps=0.0, dropout=0.1
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✓ ProteinMPNN loaded")

    # Initialize generators
    print_section("STEP 2: Initialize Generators")
    gamma_gen = GammaGenerator(embedding_dim=128, hidden_dim=64).to(device)
    delta_gen = DeltaGenerator(embedding_dim=128, hidden_dim=64).to(device)

    print(f"✓ Generators initialized")
    print(f"  - Gamma parameters: {sum(p.numel() for p in gamma_gen.parameters())}")
    print(f"  - Delta parameters: {sum(p.numel() for p in delta_gen.parameters())}")

    # Train
    print_section("STEP 3: Train with Surrogate Losses")

    trained_gamma, trained_delta, history = train_generators_simplified(
        model=model,
        gamma_gen=gamma_gen,
        delta_gen=delta_gen,
        epochs=100,
        lr=1e-3,
        device=device,
        save_path='checkpoints'
    )

    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  TRAINING COMPLETE - Run evaluation script next".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")

    print("Next step:")
    print("  python evaluate_trained_generators.py")


if __name__ == "__main__":
    main()
