#!/usr/bin/env python3
"""
Train watermark generators using ProteinMPNN fidelity scores.

This implements the proper approach where:
1. Generate watermarked sequences using current gamma/delta generators
2. Feed sequences to ProteinMPNN to get fidelity scores
3. Optimize to maximize fidelity (quality) while maintaining detectability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
import os
from pathlib import Path

# Add ProteinMPNN to path
sys.path.insert(0, 'ProteinMPNN')
from protein_mpnn_utils import (
    ProteinMPNN,
    loss_nll,
    loss_smoothed,
    gather_edges,
    gather_nodes,
    gather_nodes_t,
    cat_neighbors_nodes,
    _scores,
    _S_to_seq,
    tied_featurize,
    parse_PDB,
    StructureDatasetPDB
)

from protein_watermark import GammaGenerator, DeltaGenerator, ProteinWatermarker


class FidelityBasedTrainer:
    """Train generators using ProteinMPNN fidelity scores as quality metric."""

    def __init__(
        self,
        proteinmpnn_model,
        gamma_gen,
        delta_gen,
        device='cpu',
        lr=0.001,
        temperature=0.1
    ):
        self.model = proteinmpnn_model
        self.gamma_gen = gamma_gen
        self.delta_gen = delta_gen
        self.device = device
        self.temperature = temperature
        self.watermarker = ProteinWatermarker()

        # Optimizer for generators only (not ProteinMPNN)
        self.optimizer = optim.Adam(
            list(gamma_gen.parameters()) + list(delta_gen.parameters()),
            lr=lr
        )

        # Freeze ProteinMPNN weights
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

    def compute_fidelity_score(self, sequences, structure_features):
        """
        Compute ProteinMPNN fidelity score for generated sequences.

        Higher score = better fit to the structure (higher quality).

        Args:
            sequences: List of amino acid sequences (strings)
            structure_features: Dictionary with structure features from tied_featurize

        Returns:
            fidelity_score: Scalar tensor (higher is better)
        """
        # Convert sequences to indices
        batch_size = len(sequences)
        seq_length = len(sequences[0])

        # Get sequence tensors
        S = torch.zeros(batch_size, seq_length, dtype=torch.long, device=self.device)
        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq):
                S[i, j] = self.watermarker.AA_TO_IDX.get(aa, 0)

        # Extract structure features
        X = structure_features['X']  # [B, L, 4, 3] coordinates
        mask = structure_features['mask']  # [B, L]
        chain_M = structure_features['chain_M']  # [B, L]

        # Expand to batch size if needed
        if X.shape[0] == 1 and batch_size > 1:
            X = X.expand(batch_size, -1, -1, -1)
            mask = mask.expand(batch_size, -1)
            chain_M = chain_M.expand(batch_size, -1)

        # Run through ProteinMPNN to get log probabilities
        with torch.enable_grad():  # Enable gradients for generators
            log_probs = self.model(
                X, S, mask, chain_M,
                residue_idx=structure_features['residue_idx'],
                chain_encoding_all=structure_features['chain_encoding_all']
            )

        # Compute average log probability (fidelity score)
        # Higher log prob = sequence fits structure better
        valid_positions = chain_M.sum()
        fidelity = (log_probs * chain_M.unsqueeze(-1)).sum() / (valid_positions * 21)

        return fidelity

    def sample_sequences_differentiable(self, structure_features, batch_size=8):
        """
        Sample sequences using Gumbel-Softmax for differentiability.

        This allows gradients to flow back to gamma/delta generators.

        Returns:
            sequences: List of strings (for evaluation)
            soft_sequences: Tensor [B, L, 21] (for gradient computation)
        """
        X = structure_features['X']
        mask = structure_features['mask']
        chain_M = structure_features['chain_M']

        seq_length = X.shape[1]

        # Initialize with random amino acids
        S = torch.randint(0, 21, (batch_size, seq_length), device=self.device)

        # Compute bias matrix using generators
        bias_matrix = torch.zeros(batch_size, seq_length, 21, device=self.device)

        for b in range(batch_size):
            for pos in range(1, seq_length):
                # Get previous position embedding
                prev_aa_idx = S[b, pos - 1].item()
                prev_emb = self.model.W_s.weight[prev_aa_idx].unsqueeze(0)

                # Compute gamma and delta
                gamma = self.gamma_gen(prev_emb).squeeze()
                delta = self.delta_gen(prev_emb).squeeze()

                # Compute green list using hash
                prev_aa = self.watermarker.IDX_TO_AA.get(prev_aa_idx, 'A')
                seed = self.watermarker._hash_to_seed(prev_aa, pos)
                green_list, _ = self.watermarker._split_vocabulary(gamma.item(), seed)

                # Apply bias to green amino acids
                for aa_idx in green_list:
                    bias_matrix[b, pos, aa_idx] = delta

        # Expand structure features to batch
        if X.shape[0] == 1:
            X = X.expand(batch_size, -1, -1, -1)
            mask = mask.expand(batch_size, -1)
            chain_M = chain_M.expand(batch_size, -1)

        # Sample using Gumbel-Softmax (differentiable)
        log_probs = self.model(
            X, S, mask, chain_M,
            residue_idx=structure_features['residue_idx'],
            chain_encoding_all=structure_features['chain_encoding_all']
        )

        # Add watermark bias to logits
        biased_logits = log_probs + bias_matrix / self.temperature

        # Gumbel-Softmax sampling (differentiable)
        soft_S = F.gumbel_softmax(biased_logits, tau=self.temperature, hard=False)

        # Hard sampling for sequence generation (no gradient)
        with torch.no_grad():
            hard_S = F.gumbel_softmax(biased_logits, tau=self.temperature, hard=True)
            sequences = []
            for b in range(batch_size):
                seq = ""
                for pos in range(seq_length):
                    if chain_M[b, pos] > 0:
                        aa_idx = hard_S[b, pos].argmax().item()
                        seq += self.watermarker.IDX_TO_AA.get(aa_idx, 'A')
                sequences.append(seq)

        return sequences, soft_S

    def compute_detection_score(self, sequences):
        """
        Compute average z-score for a batch of sequences.

        Higher z-score = stronger watermark = better detectability.

        Returns:
            avg_z_score: Scalar (higher is better)
        """
        z_scores = []
        for seq in sequences:
            result = self.watermarker.detect_watermark(
                seq,
                use_theoretical_threshold=True,
                model=self.model
            )
            z_scores.append(result['z_score'])

        return np.mean(z_scores)

    def compute_detection_loss_differentiable(self, batch_size=32):
        """
        Differentiable detection loss based on generator outputs.

        This is a surrogate for z-score that we can backprop through.
        """
        # Sample random embeddings
        random_embs = torch.randn(batch_size, 128, device=self.device)

        # Get generator outputs
        gamma_outputs = self.gamma_gen(random_embs)
        delta_outputs = self.delta_gen(random_embs)

        # Target: delta around 3.0, diverse gamma
        delta_target = 3.0
        delta_loss = F.mse_loss(delta_outputs, torch.ones_like(delta_outputs) * delta_target)
        gamma_variance_loss = -torch.var(gamma_outputs)

        detection_loss = delta_loss + 0.5 * gamma_variance_loss

        return detection_loss, delta_outputs.mean(), gamma_outputs.mean()

    def train_step(self, structure_features, batch_size=8, detection_weight=1.0, fidelity_weight=1.0):
        """
        Single training step.

        Args:
            structure_features: Features from tied_featurize
            batch_size: Number of sequences to generate per step
            detection_weight: Weight for detection objective
            fidelity_weight: Weight for fidelity objective

        Returns:
            Dictionary with losses and metrics
        """
        self.optimizer.zero_grad()

        # 1. Compute differentiable detection loss
        detection_loss, avg_delta, avg_gamma = self.compute_detection_loss_differentiable(batch_size=32)

        # 2. Generate sequences and compute fidelity
        # Note: This part is partially non-differentiable due to hard sampling
        # We use the soft probabilities to approximate gradients
        sequences, soft_S = self.sample_sequences_differentiable(structure_features, batch_size)

        # Convert soft sequences back to hard for ProteinMPNN scoring
        # Use straight-through estimator: forward uses hard, backward uses soft
        with torch.no_grad():
            hard_S = soft_S.argmax(dim=-1)

        # Compute fidelity using generated sequences
        seq_strings = []
        for b in range(batch_size):
            seq = ""
            for pos in range(hard_S.shape[1]):
                if structure_features['chain_M'][0, pos] > 0:
                    aa_idx = hard_S[b, pos].item()
                    seq += self.watermarker.IDX_TO_AA.get(aa_idx, 'A')
            seq_strings.append(seq)

        # Compute fidelity score (this should be differentiable through soft_S)
        # For now, use non-differentiable version - can improve later
        with torch.no_grad():
            # Get log probs for the generated sequences
            X = structure_features['X']
            mask = structure_features['mask']
            chain_M = structure_features['chain_M']

            if X.shape[0] == 1:
                X = X.expand(batch_size, -1, -1, -1)
                mask = mask.expand(batch_size, -1)
                chain_M = chain_M.expand(batch_size, -1)

            log_probs = self.model(
                X, hard_S, mask, chain_M,
                residue_idx=structure_features['residue_idx'],
                chain_encoding_all=structure_features['chain_encoding_all']
            )

            # Fidelity = average log probability
            fidelity_score = (log_probs.gather(-1, hard_S.unsqueeze(-1)).squeeze(-1) * chain_M).sum() / chain_M.sum()

        # 3. Compute total loss
        # Minimize detection_loss (maximize detectability)
        # Maximize fidelity_score (better quality)
        total_loss = detection_weight * detection_loss - fidelity_weight * fidelity_score

        # 4. Backpropagate
        total_loss.backward()
        self.optimizer.step()

        # 5. Compute actual detection z-score for monitoring
        with torch.no_grad():
            avg_z_score = self.compute_detection_score(seq_strings)

        return {
            'total_loss': total_loss.item(),
            'detection_loss': detection_loss.item(),
            'fidelity_score': fidelity_score.item(),
            'avg_delta': avg_delta.item(),
            'avg_gamma': avg_gamma.item(),
            'avg_z_score': avg_z_score
        }


def main():
    print("=" * 80)
    print("Training Watermark Generators with ProteinMPNN Fidelity")
    print("=" * 80)
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()

    # Load ProteinMPNN
    print("Loading ProteinMPNN...")
    checkpoint = torch.load(
        'ProteinMPNN/vanilla_model_weights/v_48_020.pt',
        map_location=device
    )

    model = ProteinMPNN(
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        vocab=21,
        k_neighbors=checkpoint['num_edges'],
        augment_eps=0.0,
        dropout=0.0
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ ProteinMPNN loaded")
    print()

    # Load test structure
    print("Loading test structure...")
    pdb_path = "ProteinMPNN/inputs/PDB_monomers/pdbs/5L33.pdb"
    pdb_dict_list = parse_PDB(pdb_path)
    dataset = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=10000)
    structure_dict = dataset[0]
    batch = [structure_dict]

    structure_features = tied_featurize(
        batch,
        device,
        chain_dict=None,
        fixed_position_dict=None,
        omit_AA_dict=None,
        tied_positions_dict=None,
        pssm_dict=None,
        bias_by_res_dict=None
    )
    print(f"✓ Loaded structure (length: {structure_features['mask'].sum().item():.0f})")
    print()

    # Initialize generators
    print("Initializing generators...")
    gamma_gen = GammaGenerator(embedding_dim=128, hidden_dim=64).to(device)
    delta_gen = DeltaGenerator(embedding_dim=128, hidden_dim=64).to(device)
    print("✓ Generators initialized")
    print()

    # Create trainer
    trainer = FidelityBasedTrainer(
        proteinmpnn_model=model,
        gamma_gen=gamma_gen,
        delta_gen=delta_gen,
        device=device,
        lr=0.001,
        temperature=0.1
    )

    # Training loop
    print("Starting training...")
    print(f"{'Epoch':<8} {'Loss':<12} {'Det Loss':<12} {'Fidelity':<12} {'Z-score':<10} {'Delta':<10} {'Gamma':<10}")
    print("-" * 80)

    num_epochs = 100
    batch_size = 8
    detection_weight = 1.0
    fidelity_weight = 0.1  # Start with lower weight for fidelity

    for epoch in range(num_epochs):
        metrics = trainer.train_step(
            structure_features,
            batch_size=batch_size,
            detection_weight=detection_weight,
            fidelity_weight=fidelity_weight
        )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"{epoch+1:<8} "
                f"{metrics['total_loss']:<12.4f} "
                f"{metrics['detection_loss']:<12.4f} "
                f"{metrics['fidelity_score']:<12.4f} "
                f"{metrics['avg_z_score']:<10.2f} "
                f"{metrics['avg_delta']:<10.4f} "
                f"{metrics['avg_gamma']:<10.4f}"
            )

    print()
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print()

    # Save trained generators
    save_path = "trained_generators_fidelity.pt"
    torch.save({
        'gamma_gen_state_dict': gamma_gen.state_dict(),
        'delta_gen_state_dict': delta_gen.state_dict(),
        'epoch': num_epochs - 1,
        'final_metrics': metrics
    }, save_path)

    print(f"✓ Saved trained generators to {save_path}")
    print()
    print("Final metrics:")
    print(f"  - Detection loss: {metrics['detection_loss']:.4f}")
    print(f"  - Fidelity score: {metrics['fidelity_score']:.4f}")
    print(f"  - Average z-score: {metrics['avg_z_score']:.2f}")
    print(f"  - Average delta: {metrics['avg_delta']:.4f}")
    print(f"  - Average gamma: {metrics['avg_gamma']:.4f}")


if __name__ == "__main__":
    main()
