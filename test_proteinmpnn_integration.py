"""
Real ProteinMPNN Integration Test with Watermark Generation
This test actually loads ProteinMPNN and generates watermarked protein sequences
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add ProteinMPNN to path
sys.path.insert(0, str(Path(__file__).parent / "ProteinMPNN"))

from protein_watermark import GammaGenerator, DeltaGenerator, ProteinWatermarker
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


class WatermarkedProteinMPNN:
    """Wrapper that adds watermarking to ProteinMPNN sampling"""

    def __init__(self, model, gamma_gen, delta_gen, secret_key):
        self.model = model
        self.gamma_gen = gamma_gen
        self.delta_gen = delta_gen
        self.watermarker = ProteinWatermarker(gamma_gen, delta_gen, secret_key)
        self.alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

    def sample_watermarked(self, X, randn, S_true, chain_mask, chain_encoding_all,
                          residue_idx, mask, temperature=0.1, chain_M_pos=None):
        """
        Generate sequence with watermarking by modifying ProteinMPNN's forward pass
        Uses a simpler approach: get log_probs for all positions, modify, then sample
        """
        device = X.device

        # Get unconditional probabilities from ProteinMPNN for all positions
        # This gives us the base distribution
        with torch.no_grad():
            log_probs = self.model(X, S_true, mask, chain_mask, residue_idx,
                                   chain_encoding_all, randn, use_input_decoding_order=False)
            # log_probs shape: [batch, length, vocab=21]

        # Now sample autoregressively with watermarking
        N_batch, N_nodes = X.size(0), X.size(1)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)

        # Determine decoding order
        chain_mask_combined = chain_mask * chain_M_pos * mask
        decoding_order = torch.argsort((chain_mask_combined + 0.0001) * torch.abs(randn))

        watermark_stats = {'gamma': [], 'delta': [], 'green_lists': [], 'positions': []}

        # Sample autoregressively
        for t_ in range(N_nodes):
            t = decoding_order[0, t_]  # Position index

            mask_val = mask[0, t].item()
            if mask_val == 0:
                # Skip masked positions
                S[0, t] = S_true[0, t]
                continue

            # Get logits for this position from ProteinMPNN
            logits = log_probs[0, t, :] / 0.01  # Convert log_probs back to logits (approx)

            # Apply watermark if not first position
            if t_ > 0:
                # Get previous amino acid (in decoding order)
                prev_t = decoding_order[0, t_ - 1]
                prev_aa_idx = S[0, prev_t].item()
                prev_aa = self.alphabet[prev_aa_idx]

                # Get embedding
                prev_emb = self.model.W_s.weight[prev_aa_idx]

                # Generate gamma and delta
                with torch.no_grad():
                    gamma = self.gamma_gen(prev_emb.unsqueeze(0)).item()
                    delta = self.delta_gen(prev_emb.unsqueeze(0)).item()

                # Split vocabulary
                seed = self.watermarker._hash_to_seed(prev_aa)
                green_list, red_list = self.watermarker._split_vocabulary(gamma, seed)

                # Apply watermark: add delta to green amino acids
                for aa_idx in green_list:
                    logits[aa_idx] += delta

                # Store stats
                watermark_stats['gamma'].append(gamma)
                watermark_stats['delta'].append(delta)
                watermark_stats['green_lists'].append(green_list)
                watermark_stats['positions'].append(t.item())

            # Sample from watermarked distribution
            probs = F.softmax(logits / temperature, dim=-1)
            aa_idx = torch.multinomial(probs, 1).item()
            S[0, t] = aa_idx

        # Convert to sequence string
        sequence = ''.join([self.alphabet[S[0, i].item()] for i in range(N_nodes) if mask[0, i] == 1])

        return sequence, watermark_stats


def test_real_proteinmpnn_watermarking():
    """Test actual ProteinMPNN sequence generation with watermarking"""
    print_section("REAL PROTEINMPNN WATERMARKING TEST")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # 1. Load ProteinMPNN model
    print_section("STEP 1: Load ProteinMPNN Model")
    model_path = "ProteinMPNN/vanilla_model_weights/v_48_020.pt"

    if not os.path.exists(model_path):
        print(f"✗ Model weights not found: {model_path}")
        print("\nTo download:")
        print(f"  mkdir -p ProteinMPNN/vanilla_model_weights")
        print(f"  wget https://files.ipd.uw.edu/pub/ProteinMPNN/model_weights/v_48_020.pt \\")
        print(f"       -O {model_path}")
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
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 2. Load PDB structure
    print_section("STEP 2: Load PDB Structure")
    pdb_path = "ProteinMPNN/inputs/PDB_monomers/pdbs/5L33.pdb"

    if not os.path.exists(pdb_path):
        print(f"✗ PDB file not found: {pdb_path}")
        return False

    pdb_dict_list = parse_PDB(pdb_path)
    dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=10000)

    # Get single structure and wrap in list for batch processing
    structure_dict = dataset_valid[0]
    batch = [structure_dict]

    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, \
        visible_list_list, masked_list_list, masked_chain_length_list_list, \
        chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, \
        pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(
        batch, device, None, None, None, None, None, None, ca_only=False)

    pdb_id = structure_dict['name']
    native_seq = structure_dict['seq']

    print(f"✓ Structure loaded: {pdb_id}")
    print(f"  - Length: {len(native_seq)}")
    print(f"  - Native sequence: {native_seq[:60]}...")

    # 3. Initialize watermarking generators
    print_section("STEP 3: Initialize Watermarking Generators")

    gamma_gen = GammaGenerator(embedding_dim=128, hidden_dim=64).to(device)
    delta_gen = DeltaGenerator(embedding_dim=128, hidden_dim=64).to(device)

    print(f"✓ Generators initialized")
    print(f"  - Gamma params: {sum(p.numel() for p in gamma_gen.parameters())}")
    print(f"  - Delta params: {sum(p.numel() for p in delta_gen.parameters())}")

    watermarker = ProteinWatermarker(gamma_gen, delta_gen, secret_key="proteinmpnn_integration_test")
    wm_model = WatermarkedProteinMPNN(model, gamma_gen, delta_gen, watermarker.secret_key)

    # 4. Generate watermarked sequences
    print_section("STEP 4: Generate Watermarked Sequences")

    n_sequences = 5
    temperature = 0.1
    omit_AAs_np = np.zeros(21)

    watermarked_sequences = []
    baseline_sequences = []

    print(f"\nGenerating {n_sequences} sequences...")

    for i in range(n_sequences):
        # Set random seed for reproducibility
        randn = torch.randn(chain_M.shape, device=device)

        # Generate watermarked sequence
        wm_seq, wm_stats = wm_model.sample_watermarked(
            X, randn, S, chain_M, chain_encoding_all, residue_idx, mask,
            temperature=temperature, chain_M_pos=chain_M_pos
        )

        watermarked_sequences.append((wm_seq, wm_stats))

        # Generate baseline (no watermark) for comparison
        with torch.no_grad():
            S_sample = model.sample(
                X, randn, S, chain_M, chain_encoding_all, residue_idx,
                mask=mask, temperature=temperature, omit_AAs_np=omit_AAs_np,
                bias_AAs_np=np.zeros(21), chain_M_pos=chain_M_pos,
                omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef,
                pssm_bias=pssm_bias, pssm_multi=0.0, pssm_log_odds_flag=False,
                pssm_log_odds_mask=None, pssm_bias_flag=False,
                bias_by_res=bias_by_res_all
            )

        alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        baseline_seq = ''.join([alphabet[S_sample[0, i].item()] for i in range(len(native_seq)) if mask[0, i] == 1])
        baseline_sequences.append(baseline_seq)

        print(f"  Sequence {i + 1}:")
        print(f"    Watermarked: {wm_seq[:60]}...")
        print(f"    Baseline:    {baseline_seq[:60]}...")
        if len(wm_stats['gamma']) > 0:
            print(f"    Avg gamma: {np.mean(wm_stats['gamma']):.3f}, Avg delta: {np.mean(wm_stats['delta']):.3f}")

    # 5. Detect watermarks
    print_section("STEP 5: Watermark Detection")

    print(f"\nDetecting watermarks in generated sequences:")
    print(f"\n{'Type':<15} {'Seq#':<8} {'Z-score':<12} {'P-value':<12} {'Watermarked'}")
    print("-" * 80)

    wm_detections = []
    baseline_detections = []

    for i, (wm_seq, wm_stats) in enumerate(watermarked_sequences):
        result = watermarker.detect_watermark(wm_seq)
        wm_detections.append(result)
        print(f"{'Watermarked':<15} {i + 1:<8} {result['z_score']:<12.4f} {result['p_value']:<12.6f} {result['is_watermarked']}")

    print()

    for i, baseline_seq in enumerate(baseline_sequences):
        result = watermarker.detect_watermark(baseline_seq)
        baseline_detections.append(result)
        print(f"{'Baseline':<15} {i + 1:<8} {result['z_score']:<12.4f} {result['p_value']:<12.6f} {result['is_watermarked']}")

    # Also test native sequence
    native_result = watermarker.detect_watermark(native_seq)
    print(f"{'Native':<15} {'-':<8} {native_result['z_score']:<12.4f} {native_result['p_value']:<12.6f} {native_result['is_watermarked']}")

    # 6. Statistics and analysis
    print_section("STEP 6: Analysis")

    wm_z_scores = [d['z_score'] for d in wm_detections]
    baseline_z_scores = [d['z_score'] for d in baseline_detections]

    wm_detected = sum(1 for d in wm_detections if d['is_watermarked'])
    baseline_detected = sum(1 for d in baseline_detections if d['is_watermarked'])

    print(f"\nWatermarked sequences:")
    print(f"  - Mean Z-score: {np.mean(wm_z_scores):.4f} ± {np.std(wm_z_scores):.4f}")
    print(f"  - Detection rate: {wm_detected}/{n_sequences} ({100 * wm_detected / n_sequences:.1f}%)")

    print(f"\nBaseline sequences:")
    print(f"  - Mean Z-score: {np.mean(baseline_z_scores):.4f} ± {np.std(baseline_z_scores):.4f}")
    print(f"  - False positive rate: {baseline_detected}/{n_sequences} ({100 * baseline_detected / n_sequences:.1f}%)")

    # 7. Sequence similarity
    print_section("STEP 7: Sequence Similarity")

    print(f"\nComparing watermarked vs baseline sequences:")
    for i in range(n_sequences):
        wm_seq = watermarked_sequences[i][0]
        base_seq = baseline_sequences[i]

        matches = sum(1 for a, b in zip(wm_seq, base_seq) if a == b)
        similarity = matches / len(wm_seq)

        print(f"  Pair {i + 1}: {similarity * 100:.1f}% identical ({matches}/{len(wm_seq)} residues)")

    # Summary
    print_section("TEST SUMMARY")

    print("\n✓ INTEGRATION TEST COMPLETED!")
    print("\nKey Results:")
    print(f"  1. Generated {n_sequences} watermarked protein sequences")
    print(f"  2. Generated {n_sequences} baseline sequences for comparison")
    print(f"  3. Watermarked detection rate: {100 * wm_detected / n_sequences:.1f}%")
    print(f"  4. Baseline false positive rate: {100 * baseline_detected / n_sequences:.1f}%")
    print(f"  5. Mean watermarked Z-score: {np.mean(wm_z_scores):.4f}")
    print(f"  6. Mean baseline Z-score: {np.mean(baseline_z_scores):.4f}")

    if wm_detected > baseline_detected:
        print(f"\n✓ Watermark detection is working (more watermarked sequences detected)!")
    else:
        print(f"\n⚠ Note: Generators need training to improve detection")
        print(f"  Currently using random untrained generators")

    print("\n" + "=" * 80)

    return True


def main():
    """Run the test"""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  PROTEINMPNN WATERMARKING INTEGRATION TEST".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    torch.manual_seed(42)
    np.random.seed(42)

    try:
        success = test_real_proteinmpnn_watermarking()

        if success:
            print("\n" + "█" * 80)
            print("█" + " " * 78 + "█")
            print("█" + "  STATUS: INTEGRATION TEST PASSED".center(78) + "█")
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
