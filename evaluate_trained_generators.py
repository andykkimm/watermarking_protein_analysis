"""
Evaluate Trained Watermark Generators
Tests detection rates after MGDA training
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path

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


# Import generator classes from training script
from train_watermark_generators import GammaGenerator, DeltaGenerator


def evaluate_generators(model, gamma_gen, delta_gen, test_structure, device, n_sequences=20):
    """Evaluate trained generators on test structure"""

    watermarker = ProteinWatermarker(gamma_gen, delta_gen, secret_key="evaluation")
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

    X = test_structure['X']
    S = test_structure['S']
    mask = test_structure['mask']
    chain_M = test_structure['chain_M']
    chain_encoding_all = test_structure['chain_encoding_all']
    residue_idx = test_structure['residue_idx']
    chain_M_pos = test_structure['chain_M_pos']
    bias_by_res_all = test_structure['bias_by_res_all']
    seq_length = test_structure['lengths']

    # Compute watermark bias
    bias_matrix = torch.zeros(1, seq_length, 21, device=device)

    print("Computing watermark bias matrix...")
    with torch.no_grad():
        for pos in range(1, seq_length):
            position_bias = torch.zeros(21, device=device)

            for prev_aa_idx in range(21):
                prev_aa = alphabet[prev_aa_idx]
                prev_emb = model.W_s.weight[prev_aa_idx]

                gamma = gamma_gen(prev_emb.unsqueeze(0)).item()
                delta = delta_gen(prev_emb.unsqueeze(0)).item()

                seed = watermarker._hash_to_seed(prev_aa)
                green_list, _ = watermarker._split_vocabulary(gamma, seed)

                for aa_idx in green_list:
                    position_bias[aa_idx] += delta

            bias_matrix[0, pos, :] = position_bias / 21.0

    print(f"  Mean bias: {bias_matrix.mean().item():.3f}")
    print(f"  Max bias: {bias_matrix.max().item():.3f}")

    # Generate sequences
    print(f"\nGenerating {n_sequences} test sequences...")

    watermarked_sequences = []
    baseline_sequences = []
    omit_AAs_np = np.zeros(21)

    for i in range(n_sequences):
        randn = torch.randn(chain_M.shape, device=device)

        # Watermarked
        with torch.no_grad():
            output_wm = model.sample(
                X, randn, S, chain_M, chain_encoding_all, residue_idx,
                mask=mask, temperature=0.1, omit_AAs_np=omit_AAs_np,
                bias_AAs_np=np.zeros(21), chain_M_pos=chain_M_pos,
                omit_AA_mask=None, pssm_coef=None,
                pssm_bias=None, pssm_multi=0.0, pssm_log_odds_flag=False,
                pssm_log_odds_mask=None, pssm_bias_flag=False,
                bias_by_res=bias_matrix
            )

        S_wm = output_wm["S"]
        wm_seq = ''.join([alphabet[S_wm[0, i].item()] for i in range(seq_length) if mask[0, i] == 1])
        watermarked_sequences.append(wm_seq)

        # Baseline
        with torch.no_grad():
            output_base = model.sample(
                X, randn, S, chain_M, chain_encoding_all, residue_idx,
                mask=mask, temperature=0.1, omit_AAs_np=omit_AAs_np,
                bias_AAs_np=np.zeros(21), chain_M_pos=chain_M_pos,
                omit_AA_mask=None, pssm_coef=None,
                pssm_bias=None, pssm_multi=0.0, pssm_log_odds_flag=False,
                pssm_log_odds_mask=None, pssm_bias_flag=False,
                bias_by_res=bias_by_res_all
            )

        S_base = output_base["S"]
        base_seq = ''.join([alphabet[S_base[0, i].item()] for i in range(seq_length) if mask[0, i] == 1])
        baseline_sequences.append(base_seq)

    # Detect
    print("\nRunning detection...")

    wm_detections = []
    baseline_detections = []

    for wm_seq in watermarked_sequences:
        result = watermarker.detect_watermark(wm_seq, model=model)
        wm_detections.append(result)

    for base_seq in baseline_sequences:
        result = watermarker.detect_watermark(base_seq, model=model)
        baseline_detections.append(result)

    return wm_detections, baseline_detections


def main():
    """Evaluate trained generators"""

    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  EVALUATE TRAINED WATERMARK GENERATORS".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Load ProteinMPNN
    print_section("STEP 1: Load ProteinMPNN Model")
    model_path = "ProteinMPNN/vanilla_model_weights/v_48_020.pt"
    checkpoint = torch.load(model_path, map_location=device)

    model = ProteinMPNN(
        num_letters=21, node_features=128, edge_features=128, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=3, vocab=21,
        k_neighbors=48, augment_eps=0.0, dropout=0.1
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print("✓ ProteinMPNN loaded")

    # Load trained generators
    print_section("STEP 2: Load Trained Generators")

    checkpoint_path = "checkpoints/best_generators.pt"

    if not os.path.exists(checkpoint_path):
        print(f"✗ No trained model found at {checkpoint_path}")
        print("Run train_watermark_generators.py first!")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)

    gamma_gen = GammaGenerator(embedding_dim=128, hidden_dim=64).to(device)
    delta_gen = DeltaGenerator(embedding_dim=128, hidden_dim=64).to(device)

    gamma_gen.load_state_dict(checkpoint['gamma_gen_state_dict'])
    delta_gen.load_state_dict(checkpoint['delta_gen_state_dict'])

    gamma_gen.eval()
    delta_gen.eval()

    print(f"✓ Loaded trained generators")
    if 'avg_delta' in checkpoint:
        print(f"  - Training avg delta: {checkpoint['avg_delta']:.4f}")
        if 'avg_gamma' in checkpoint:
            print(f"  - Training avg gamma: {checkpoint['avg_gamma']:.4f}")
    if 'epoch' in checkpoint:
        print(f"  - From epoch: {checkpoint['epoch']}")

    # Load test structure
    print_section("STEP 3: Load Test Structure")

    pdb_path = "ProteinMPNN/inputs/PDB_monomers/pdbs/5L33.pdb"
    pdb_dict_list = parse_PDB(pdb_path)
    dataset = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=10000)

    structure_dict = dataset[0]
    batch = [structure_dict]

    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, \
        visible_list_list, masked_list_list, masked_chain_length_list_list, \
        chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, \
        pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(
        batch, device, None, None, None, None, None, None, ca_only=False)

    test_structure = {
        'X': X, 'S': S, 'mask': mask, 'lengths': lengths[0],
        'chain_M': chain_M, 'chain_encoding_all': chain_encoding_all,
        'chain_M_pos': chain_M_pos, 'residue_idx': residue_idx,
        'bias_by_res_all': bias_by_res_all
    }

    print(f"✓ Loaded test structure (length: {lengths[0]})")

    # Evaluate
    print_section("STEP 4: Evaluate Detection Performance")

    wm_detections, baseline_detections = evaluate_generators(
        model, gamma_gen, delta_gen, test_structure, device, n_sequences=20
    )

    # Analyze results
    print_section("RESULTS")

    wm_z_scores = [d['z_score'] for d in wm_detections]
    baseline_z_scores = [d['z_score'] for d in baseline_detections]

    wm_detected = sum(1 for d in wm_detections if d['is_watermarked'])
    baseline_detected = sum(1 for d in baseline_detections if d['is_watermarked'])

    print(f"\n{'Type':<15} {'Z-score':<20} {'Detection Rate'}")
    print("-" * 60)
    print(f"{'Watermarked':<15} {np.mean(wm_z_scores):.3f} ± {np.std(wm_z_scores):.3f}      "
          f"{wm_detected}/20 ({100*wm_detected/20:.0f}%)")
    print(f"{'Baseline':<15} {np.mean(baseline_z_scores):.3f} ± {np.std(baseline_z_scores):.3f}      "
          f"{baseline_detected}/20 ({100*baseline_detected/20:.0f}%)")

    z_sep = np.mean(wm_z_scores) - np.mean(baseline_z_scores)

    print(f"\n📊 Performance Metrics:")
    print(f"  - Z-score separation: {z_sep:.4f}")
    print(f"  - True Positive Rate: {100*wm_detected/20:.1f}%")
    print(f"  - False Positive Rate: {100*baseline_detected/20:.1f}%")

    # Show at different thresholds
    print(f"\n📈 Detection Rates at Different Thresholds:")
    for fpr_target in [0.01, 0.05, 0.10]:
        from scipy.stats import norm
        threshold = norm.ppf(1 - fpr_target)
        wm_det = sum(1 for z in wm_z_scores if z > threshold)
        print(f"  {int(fpr_target*100)}% FPR (threshold={threshold:.2f}): "
              f"{wm_det}/20 = {100*wm_det/20:.0f}%")

    # Success criteria
    print(f"\n" + "=" * 80)
    if wm_detected >= 16:  # 80%+
        print("✅ SUCCESS: Detection rate ≥ 80%!")
    elif wm_detected >= 12:  # 60%+
        print("✓ GOOD: Detection rate ≥ 60% - continue training for better results")
    else:
        print("⚠ MODERATE: Detection working but needs more training")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
