"""
Train Watermark Generators using MGDA
Multi-objective optimization balancing detectability vs. protein quality
"""

import torch
import torch.nn as nn
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
import torch.nn.functional as F


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
        gamma = 0.3 + 0.4 * self.sigmoid(self.fc2(x))  # Range [0.3, 0.7]
        return gamma


class DeltaGenerator(nn.Module):
    """δ-generator for training"""
    def __init__(self, embedding_dim=128, hidden_dim=64):
        super(DeltaGenerator, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.LeakyReLU(0.01)
        self.softplus = nn.Softplus()

    def forward(self, prev_aa_embedding):
        x = self.relu(self.fc1(prev_aa_embedding))
        delta = self.softplus(self.fc2(x))  # Range [0, inf]
        return delta


def generate_watermarked_sequence_for_training(
    model, X, S, mask, chain_M, chain_encoding_all, residue_idx, chain_M_pos,
    gamma_gen, delta_gen, watermarker, device
):
    """
    Generate a watermarked sequence and return sequence + statistics for loss computation
    """
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    seq_length = X.size(1)

    # Compute watermark bias matrix
    bias_matrix = torch.zeros(1, seq_length, 21, device=device)

    gammas = []
    deltas = []
    green_lists = []

    for pos in range(1, seq_length):
        position_bias = torch.zeros(21, device=device)

        for prev_aa_idx in range(21):
            prev_aa = alphabet[prev_aa_idx]
            prev_emb = model.W_s.weight[prev_aa_idx]

            # Generate gamma and delta
            gamma = gamma_gen(prev_emb.unsqueeze(0))
            delta = delta_gen(prev_emb.unsqueeze(0))

            if prev_aa_idx == 0:  # Store for first AA only
                gammas.append(gamma.item())
                deltas.append(delta.item())

                seed = watermarker._hash_to_seed(prev_aa)
                green_list, _ = watermarker._split_vocabulary(gamma.item(), seed)
                green_lists.append(green_list)

            # Add delta to green amino acids
            seed = watermarker._hash_to_seed(prev_aa)
            green_list, _ = watermarker._split_vocabulary(gamma.item(), seed)

            for aa_idx in green_list:
                position_bias[aa_idx] += delta.squeeze()

        bias_matrix[0, pos, :] = position_bias / 21.0

    # Generate sequence with watermark
    randn = torch.randn(chain_M.shape, device=device)

    with torch.no_grad():
        output = model.sample(
            X, randn, S, chain_M, chain_encoding_all, residue_idx,
            mask=mask, temperature=0.1, omit_AAs_np=np.zeros(21),
            bias_AAs_np=np.zeros(21), chain_M_pos=chain_M_pos,
            omit_AA_mask=None, pssm_coef=None,
            pssm_bias=None, pssm_multi=0.0, pssm_log_odds_flag=False,
            pssm_log_odds_mask=None, pssm_bias_flag=False,
            bias_by_res=bias_matrix
        )

    S_wm = output["S"]
    seq_wm = ''.join([alphabet[S_wm[0, i].item()] for i in range(seq_length) if mask[0, i] == 1])

    return seq_wm, gammas, deltas, green_lists


def compute_detection_loss(seq, gammas, deltas, green_lists, watermarker, model):
    """
    Detection loss: maximize z-score for detectability
    Higher z-score = more detectable watermark
    Loss = -z_score (we minimize loss, so maximize z-score)
    """
    device = next(model.parameters()).device

    T = len(seq)
    green_count_est = 0
    sum_gamma = 0
    sum_variance = 0

    for i in range(1, min(T, len(gammas) + 1)):
        prev_aa = seq[i-1]
        curr_aa = seq[i]

        if i-1 < len(gammas):
            gamma = gammas[i-1]
            green_list = green_lists[i-1]

            curr_aa_idx = watermarker.AA_TO_IDX.get(curr_aa, 0)
            if curr_aa_idx in green_list:
                green_count_est += 1

            sum_gamma += gamma
            sum_variance += gamma * (1 - gamma)

    # Compute z-score
    if sum_variance > 0:
        z_score = (green_count_est - sum_gamma) / np.sqrt(sum_variance)
    else:
        z_score = 0

    # Loss: negative z-score (maximize z-score = minimize negative z-score)
    loss_detection = -torch.tensor(z_score, device=device, requires_grad=False)

    return loss_detection, z_score


def compute_semantic_loss(seq_wm, seq_original):
    """
    Semantic loss: preserve sequence similarity
    Simple version: use sequence similarity (can upgrade to ESM embeddings later)
    """
    # Compute sequence similarity
    matches = sum(1 for a, b in zip(seq_wm, seq_original) if a == b)
    similarity = matches / len(seq_wm)

    # Loss: negative similarity (maximize similarity = minimize negative similarity)
    loss_semantic = -(similarity)

    return loss_semantic, similarity


def compute_mgda_weights(grad_detection, grad_semantic):
    """
    Compute MGDA weights to balance multiple objectives
    Based on: "Multi-Task Learning as Multi-Objective Optimization" (Sener & Koltun, NeurIPS 2018)
    """
    # Flatten gradients
    g_d = torch.cat([g.flatten() for g in grad_detection if g is not None])
    g_s = torch.cat([g.flatten() for g in grad_semantic if g is not None])

    # Compute dot products
    g_d_norm = torch.norm(g_d)
    g_s_norm = torch.norm(g_s)

    if g_d_norm == 0 or g_s_norm == 0:
        return 0.5  # Equal weights if one gradient is zero

    # Normalize
    g_d = g_d / g_d_norm
    g_s = g_s / g_s_norm

    # Dot product
    dot_product = torch.dot(g_d, g_s).item()

    # Compute optimal weight using closed form
    # If gradients are aligned (dot > 0), use average
    # If conflicting (dot < 0), find balance point
    if dot_product > 0:
        lambda_star = 0.5
    else:
        # Use projection method
        lambda_star = g_s_norm / (g_d_norm + g_s_norm + 1e-8)
        lambda_star = float(lambda_star.item()) if torch.is_tensor(lambda_star) else lambda_star

    return lambda_star


def train_generators(
    model,
    train_structures,
    gamma_gen,
    delta_gen,
    watermarker,
    epochs=50,
    lr=1e-3,
    device='cpu',
    save_path='checkpoints'
):
    """
    Train gamma and delta generators using MGDA
    """
    print_section("TRAINING CONFIGURATION")

    print(f"Training parameters:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Training structures: {len(train_structures)}")
    print(f"  - Device: {device}")

    # Create save directory
    os.makedirs(save_path, exist_ok=True)

    # Optimizers
    optimizer = optim.Adam(
        list(gamma_gen.parameters()) + list(delta_gen.parameters()),
        lr=lr
    )

    # Training history
    history = {
        'detection_loss': [],
        'semantic_loss': [],
        'z_scores': [],
        'similarities': [],
        'lambda_weights': []
    }

    best_z_score = -float('inf')

    print_section("TRAINING LOOP")

    for epoch in range(epochs):
        gamma_gen.train()
        delta_gen.train()

        epoch_detection_loss = 0
        epoch_semantic_loss = 0
        epoch_z_scores = []
        epoch_similarities = []
        epoch_lambdas = []

        # Progress bar
        pbar = tqdm(train_structures, desc=f"Epoch {epoch+1}/{epochs}")

        for structure in pbar:
            optimizer.zero_grad()

            # Generate original (baseline) sequence
            with torch.no_grad():
                randn = torch.randn(structure['chain_M'].shape, device=device)
                output_orig = model.sample(
                    structure['X'], randn, structure['S'], structure['chain_M'],
                    structure['chain_encoding_all'], structure['residue_idx'],
                    mask=structure['mask'], temperature=0.1, omit_AAs_np=np.zeros(21),
                    bias_AAs_np=np.zeros(21), chain_M_pos=structure['chain_M_pos'],
                    omit_AA_mask=None, pssm_coef=None, pssm_bias=None,
                    pssm_multi=0.0, pssm_log_odds_flag=False,
                    pssm_log_odds_mask=None, pssm_bias_flag=False,
                    bias_by_res=structure['bias_by_res_all']
                )
                alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
                S_orig = output_orig["S"]
                seq_orig = ''.join([alphabet[S_orig[0, i].item()]
                                   for i in range(structure['lengths'])
                                   if structure['mask'][0, i] == 1])

            # Generate watermarked sequence (with gradients)
            seq_wm, gammas, deltas, green_lists = generate_watermarked_sequence_for_training(
                model, structure['X'], structure['S'], structure['mask'],
                structure['chain_M'], structure['chain_encoding_all'],
                structure['residue_idx'], structure['chain_M_pos'],
                gamma_gen, delta_gen, watermarker, device
            )

            # Compute losses
            loss_d, z_score = compute_detection_loss(
                seq_wm, gammas, deltas, green_lists, watermarker, model
            )

            loss_s, similarity = compute_semantic_loss(seq_wm, seq_orig)

            # Compute gradients for both losses
            loss_d_tensor = torch.tensor(loss_d, device=device, requires_grad=True)
            loss_s_tensor = torch.tensor(loss_s, device=device, requires_grad=True)

            # Get gradients
            loss_d_tensor.backward(retain_graph=True)
            grad_d = [p.grad.clone() if p.grad is not None else None
                     for p in list(gamma_gen.parameters()) + list(delta_gen.parameters())]

            optimizer.zero_grad()
            loss_s_tensor.backward(retain_graph=True)
            grad_s = [p.grad.clone() if p.grad is not None else None
                     for p in list(gamma_gen.parameters()) + list(delta_gen.parameters())]

            # Compute MGDA weights
            lambda_star = compute_mgda_weights(grad_d, grad_s)

            # Apply combined gradient
            optimizer.zero_grad()
            for p, g_d, g_s in zip(
                list(gamma_gen.parameters()) + list(delta_gen.parameters()),
                grad_d, grad_s
            ):
                if g_d is not None and g_s is not None:
                    p.grad = lambda_star * g_d + (1 - lambda_star) * g_s

            # Update
            optimizer.step()

            # Record metrics
            epoch_detection_loss += loss_d.item() if torch.is_tensor(loss_d) else loss_d
            epoch_semantic_loss += loss_s
            epoch_z_scores.append(z_score)
            epoch_similarities.append(similarity)
            epoch_lambdas.append(lambda_star)

            # Update progress bar
            pbar.set_postfix({
                'z': f'{z_score:.2f}',
                'sim': f'{similarity:.2f}',
                'λ': f'{lambda_star:.2f}'
            })

        # Epoch statistics
        avg_z = np.mean(epoch_z_scores)
        avg_sim = np.mean(epoch_similarities)
        avg_lambda = np.mean(epoch_lambdas)

        history['detection_loss'].append(epoch_detection_loss / len(train_structures))
        history['semantic_loss'].append(epoch_semantic_loss / len(train_structures))
        history['z_scores'].append(avg_z)
        history['similarities'].append(avg_sim)
        history['lambda_weights'].append(avg_lambda)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Avg Z-score: {avg_z:.4f}")
        print(f"  Avg Similarity: {avg_sim:.4f}")
        print(f"  Avg λ: {avg_lambda:.4f}")

        # Save best model
        if avg_z > best_z_score:
            best_z_score = avg_z
            torch.save({
                'epoch': epoch,
                'gamma_gen_state_dict': gamma_gen.state_dict(),
                'delta_gen_state_dict': delta_gen.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'z_score': avg_z,
                'similarity': avg_sim,
            }, os.path.join(save_path, 'best_generators.pt'))
            print(f"  ✓ Saved best model (z={avg_z:.4f})")

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

    # Save history
    with open(os.path.join(save_path, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print_section("TRAINING COMPLETE")
    print(f"Best Z-score achieved: {best_z_score:.4f}")
    print(f"Models saved to: {save_path}/")

    return gamma_gen, delta_gen, history


def load_training_data(model, pdb_dir, max_structures=10, device='cpu'):
    """Load training structures from PDB directory"""
    print_section("LOADING TRAINING DATA")

    pdb_files = list(Path(pdb_dir).glob("*.pdb"))

    if len(pdb_files) == 0:
        print(f"✗ No PDB files found in {pdb_dir}")
        return []

    print(f"Found {len(pdb_files)} PDB files")
    print(f"Loading up to {max_structures} structures...")

    structures = []

    for pdb_file in pdb_files[:max_structures]:
        try:
            pdb_dict_list = parse_PDB(str(pdb_file))
            dataset = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=10000)

            structure_dict = dataset[0]
            batch = [structure_dict]

            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, \
                visible_list_list, masked_list_list, masked_chain_length_list_list, \
                chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, \
                pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(
                batch, device, None, None, None, None, None, None, ca_only=False)

            structures.append({
                'name': pdb_file.stem,
                'X': X,
                'S': S,
                'mask': mask,
                'lengths': lengths[0],
                'chain_M': chain_M,
                'chain_encoding_all': chain_encoding_all,
                'chain_M_pos': chain_M_pos,
                'residue_idx': residue_idx,
                'bias_by_res_all': bias_by_res_all
            })

            print(f"  ✓ Loaded {pdb_file.stem} (length: {lengths[0]})")

        except Exception as e:
            print(f"  ✗ Failed to load {pdb_file.stem}: {e}")

    print(f"\n✓ Successfully loaded {len(structures)} structures")

    return structures


def main():
    """Main training script"""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  WATERMARK GENERATOR TRAINING WITH MGDA".center(78) + "█")
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

    # Load training data
    pdb_dir = "ProteinMPNN/inputs/PDB_monomers/pdbs"
    train_structures = load_training_data(model, pdb_dir, max_structures=5, device=device)

    if len(train_structures) == 0:
        print("✗ No training data available")
        return

    # Initialize generators
    print_section("STEP 2: Initialize Generators")
    gamma_gen = GammaGenerator(embedding_dim=128, hidden_dim=64).to(device)
    delta_gen = DeltaGenerator(embedding_dim=128, hidden_dim=64).to(device)
    watermarker = ProteinWatermarker(gamma_gen, delta_gen, secret_key="training")

    print(f"✓ Generators initialized")
    print(f"  - Gamma parameters: {sum(p.numel() for p in gamma_gen.parameters())}")
    print(f"  - Delta parameters: {sum(p.numel() for p in delta_gen.parameters())}")

    # Train
    print_section("STEP 3: Train with MGDA")

    trained_gamma, trained_delta, history = train_generators(
        model=model,
        train_structures=train_structures,
        gamma_gen=gamma_gen,
        delta_gen=delta_gen,
        watermarker=watermarker,
        epochs=30,
        lr=1e-3,
        device=device,
        save_path='checkpoints'
    )

    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  TRAINING COMPLETE - Run evaluation script next".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")


if __name__ == "__main__":
    main()
