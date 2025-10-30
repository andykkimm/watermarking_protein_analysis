"""
FINAL WORKING VERSION: ProteinMPNN Watermarking with bias_by_res
Uses ProteinMPNN's built-in sampling with per-residue bias for watermarking
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
import torch.nn as nn


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


class ImprovedGammaGenerator(nn.Module):
    """Improved γ-generator with better initialization"""
    def __init__(self, embedding_dim=128, hidden_dim=64):
        super(ImprovedGammaGenerator, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.LeakyReLU(0.01)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_normal_(self.fc1.weight, gain=2.0)
        nn.init.xavier_normal_(self.fc2.weight, gain=2.0)
        nn.init.constant_(self.fc2.bias, 0.5)

    def forward(self, prev_aa_embedding):
        x = self.relu(self.fc1(prev_aa_embedding))
        gamma = 0.3 + 0.4 * self.sigmoid(self.fc2(x))
        return gamma


class ImprovedDeltaGenerator(nn.Module):
    """Improved δ-generator with stronger watermark signal"""
    def __init__(self, embedding_dim=128, hidden_dim=64):
        super(ImprovedDeltaGenerator, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.LeakyReLU(0.01)
        self.softplus = nn.Softplus()

        nn.init.xavier_normal_(self.fc1.weight, gain=2.0)
        nn.init.xavier_normal_(self.fc2.weight, gain=2.0)
        nn.init.constant_(self.fc2.bias, 3.0)  # Even higher bias

    def forward(self, prev_aa_embedding):
        x = self.relu(self.fc1(prev_aa_embedding))
        delta = 2.0 + self.softplus(self.fc2(x))  # Range [2, 6+]
        return delta


def compute_watermark_bias_matrix(model, gamma_gen, delta_gen, watermarker, seq_length, device):
    """
    Pre-compute watermark bias for each position based on each possible previous amino acid.
    Returns: bias_matrix of shape (seq_length, 21) for bias_by_res parameter
    """
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    bias_matrix = torch.zeros(1, seq_length, 21, device=device)

    with torch.no_grad():
        for pos in range(1, seq_length):  # Start from 1, position 0 has no watermark
            # Average over all possible previous amino acids
            position_bias = torch.zeros(21, device=device)

            for prev_aa_idx in range(21):
                prev_aa = alphabet[prev_aa_idx]
                prev_emb = model.W_s.weight[prev_aa_idx]

                # Generate gamma and delta
                gamma = gamma_gen(prev_emb.unsqueeze(0)).item()
                delta = delta_gen(prev_emb.unsqueeze(0)).item()

                # Split vocabulary
                seed = watermarker._hash_to_seed(prev_aa)
                green_list, red_list = watermarker._split_vocabulary(gamma, seed)

                # Add delta to green amino acids
                for aa_idx in green_list:
                    position_bias[aa_idx] += delta

            # Average bias across all possible previous AAs
            bias_matrix[0, pos, :] = position_bias / 21.0

    return bias_matrix


def test_final_watermarking():
    """Test with proper bias_by_res integration"""
    print_section("FINAL PROTEINMPNN WATERMARKING TEST (Using bias_by_res)")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # 1. Load ProteinMPNN model
    print_section("STEP 1: Load ProteinMPNN Model")
    model_path = "ProteinMPNN/vanilla_model_weights/v_48_020.pt"

    if not os.path.exists(model_path):
        print(f"✗ Model weights not found: {model_path}")
        return False

    checkpoint = torch.load(model_path, map_location=device)

    model = ProteinMPNN(
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        vocab=21,
        k_neighbors=48,
        augment_eps=0.0,
        dropout=0.1
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✓ ProteinMPNN model loaded")

    # 2. Load PDB structure
    print_section("STEP 2: Load PDB Structure")
    pdb_path = "ProteinMPNN/inputs/PDB_monomers/pdbs/5L33.pdb"

    if not os.path.exists(pdb_path):
        print(f"✗ PDB file not found: {pdb_path}")
        return False

    pdb_dict_list = parse_PDB(pdb_path)
    dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=10000)

    structure_dict = dataset_valid[0]
    batch = [structure_dict]

    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, \
        visible_list_list, masked_list_list, masked_chain_length_list_list, \
        chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, \
        pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(
        batch, device, None, None, None, None, None, None, ca_only=False)

    pdb_id = structure_dict['name']
    native_seq = structure_dict['seq']
    seq_length = lengths[0]

    print(f"✓ Structure loaded: {pdb_id}")
    print(f"  - Length: {seq_length}")

    # 3. Initialize generators
    print_section("STEP 3: Initialize Watermarking Generators")

    gamma_gen = ImprovedGammaGenerator(embedding_dim=128, hidden_dim=64).to(device)
    delta_gen = ImprovedDeltaGenerator(embedding_dim=128, hidden_dim=64).to(device)

    print(f"✓ Improved generators initialized")
    print(f"  - Delta range: [2.0, 6.0+]  (very strong)")

    watermarker = ProteinWatermarker(gamma_gen, delta_gen, secret_key="final_test")

    # 4. Compute watermark bias matrix
    print_section("STEP 4: Compute Watermark Bias Matrix")

    watermark_bias = compute_watermark_bias_matrix(
        model, gamma_gen, delta_gen, watermarker, seq_length, device
    )

    print(f"✓ Watermark bias matrix computed")
    print(f"  - Shape: {watermark_bias.shape}")
    print(f"  - Mean bias: {watermark_bias.mean().item():.3f}")
    print(f"  - Max bias: {watermark_bias.max().item():.3f}")

    # 5. Generate sequences
    print_section("STEP 5: Generate Sequences")

    n_sequences = 10
    temperature = 0.1
    omit_AAs_np = np.zeros(21)

    watermarked_sequences = []
    baseline_sequences = []

    print(f"\nGenerating {n_sequences} sequences...")

    for i in range(n_sequences):
        randn = torch.randn(chain_M.shape, device=device)

        # Generate watermarked sequence WITH bias
        with torch.no_grad():
            output_wm = model.sample(
                X, randn, S, chain_M, chain_encoding_all, residue_idx,
                mask=mask, temperature=temperature, omit_AAs_np=omit_AAs_np,
                bias_AAs_np=np.zeros(21), chain_M_pos=chain_M_pos,
                omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef,
                pssm_bias=pssm_bias, pssm_multi=0.0, pssm_log_odds_flag=False,
                pssm_log_odds_mask=None, pssm_bias_flag=False,
                bias_by_res=watermark_bias  # ← WATERMARK APPLIED HERE!
            )

        alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        S_wm = output_wm["S"]
        wm_seq = ''.join([alphabet[S_wm[0, i].item()] for i in range(seq_length)])
        watermarked_sequences.append(wm_seq)

        # Generate baseline WITHOUT bias
        with torch.no_grad():
            output_base = model.sample(
                X, randn, S, chain_M, chain_encoding_all, residue_idx,
                mask=mask, temperature=temperature, omit_AAs_np=omit_AAs_np,
                bias_AAs_np=np.zeros(21), chain_M_pos=chain_M_pos,
                omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef,
                pssm_bias=pssm_bias, pssm_multi=0.0, pssm_log_odds_flag=False,
                pssm_log_odds_mask=None, pssm_bias_flag=False,
                bias_by_res=bias_by_res_all  # No watermark
            )

        S_base = output_base["S"]
        base_seq = ''.join([alphabet[S_base[0, i].item()] for i in range(seq_length)])
        baseline_sequences.append(base_seq)

        print(f"  Sequence {i + 1}: Generated")

    # 6. Detect watermarks
    print_section("STEP 6: Watermark Detection")

    print(f"\nDetecting watermarks:")
    print(f"\n{'Type':<15} {'Seq#':<8} {'Z-score':<12} {'P-value':<12} {'Detected'}")
    print("-" * 80)

    wm_detections = []
    baseline_detections = []

    for i, wm_seq in enumerate(watermarked_sequences):
        result = watermarker.detect_watermark(wm_seq, model=model)
        wm_detections.append(result)
        status = "✓" if result['is_watermarked'] else "✗"
        print(f"{'Watermarked':<15} {i + 1:<8} {result['z_score']:<12.4f} {result['p_value']:<12.6f} {status}")

    print()

    for i, baseline_seq in enumerate(baseline_sequences):
        result = watermarker.detect_watermark(baseline_seq, model=model)
        baseline_detections.append(result)
        status = "✓" if result['is_watermarked'] else "✗"
        print(f"{'Baseline':<15} {i + 1:<8} {result['z_score']:<12.4f} {result['p_value']:<12.6f} {status}")

    native_result = watermarker.detect_watermark(native_seq, model=model)
    status = "✓" if native_result['is_watermarked'] else "✗"
    print(f"{'Native':<15} {'-':<8} {native_result['z_score']:<12.4f} {native_result['p_value']:<12.6f} {status}")

    # 7. Analysis
    print_section("STEP 7: RESULTS ANALYSIS")

    wm_z_scores = [d['z_score'] for d in wm_detections]
    baseline_z_scores = [d['z_score'] for d in baseline_detections]

    wm_detected = sum(1 for d in wm_detections if d['is_watermarked'])
    baseline_detected = sum(1 for d in baseline_detections if d['is_watermarked'])

    print(f"\n📊 Watermarked Sequences:")
    print(f"  - Mean Z-score: {np.mean(wm_z_scores):.4f} ± {np.std(wm_z_scores):.4f}")
    print(f"  - Detection rate: {wm_detected}/{n_sequences} ({100 * wm_detected / n_sequences:.1f}%)")

    print(f"\n📊 Baseline Sequences:")
    print(f"  - Mean Z-score: {np.mean(baseline_z_scores):.4f} ± {np.std(baseline_z_scores):.4f}")
    print(f"  - False positive rate: {baseline_detected}/{n_sequences} ({100 * baseline_detected / n_sequences:.1f}%)")

    z_score_separation = np.mean(wm_z_scores) - np.mean(baseline_z_scores)

    print(f"\n📈 Performance Metrics:")
    print(f"  - Z-score separation: {z_score_separation:.4f}")
    print(f"  - True Positive Rate: {100 * wm_detected / n_sequences:.1f}%")
    print(f"  - False Positive Rate: {100 * baseline_detected / n_sequences:.1f}%)")

    # Summary
    print_section("FINAL SUMMARY")

    print("\n✓ TEST COMPLETED!")

    if wm_detected > 0:
        print(f"\n✅ SUCCESS: Watermark detection is working!")
        print(f"   - {wm_detected}/{n_sequences} watermarked sequences detected")
        print(f"   - Z-score separation: {z_score_separation:.2f}")
    else:
        print(f"\n⚠️  Still need improvement")
        print(f"   - The bias_by_res approach may need stronger signals")
        print(f"   - Or full training of generators is required")

    print("\n" + "=" * 80)

    return True


def main():
    """Run the test"""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  FINAL PROTEINMPNN WATERMARKING TEST".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    torch.manual_seed(42)
    np.random.seed(42)

    try:
        success = test_final_watermarking()

        if success:
            print("\n" + "█" * 80)
            print("█" + " " * 78 + "█")
            print("█" + "  STATUS: TEST COMPLETED".center(78) + "█")
            print("█" + " " * 78 + "█")
            print("█" * 80 + "\n")
            return 0
        else:
            return 1

    except Exception as e:
        print_section("TEST FAILED")
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
