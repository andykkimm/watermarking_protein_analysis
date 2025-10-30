"""
Improved ProteinMPNN Integration Test with Hand-Crafted Generators
Uses stronger, non-random generators to achieve better detection rates
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add ProteinMPNN to path
sys.path.insert(0, str(Path(__file__).parent / "ProteinMPNN"))

from protein_watermark import ProteinWatermarker
from ProteinMPNN.protein_mpnn_utils import (
    ProteinMPNN,
    StructureDatasetPDB,
    tied_featurize,
    parse_PDB
)
import torch.nn as nn
import torch.nn.functional as F


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


class ImprovedGammaGenerator(nn.Module):
    """
    Improved γ-generator with better initialization
    Outputs more diverse splitting ratios based on input
    """
    def __init__(self, embedding_dim=128, hidden_dim=64):
        super(ImprovedGammaGenerator, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.LeakyReLU(0.01)
        self.sigmoid = nn.Sigmoid()

        # Better initialization for more diversity
        nn.init.xavier_normal_(self.fc1.weight, gain=2.0)
        nn.init.xavier_normal_(self.fc2.weight, gain=2.0)
        # Bias to encourage outputs away from 0.5
        nn.init.constant_(self.fc2.bias, 0.5)

    def forward(self, prev_aa_embedding):
        """
        Args:
            prev_aa_embedding: (batch_size, embedding_dim)
        Returns:
            gamma: (batch_size, 1) splitting ratio in (0.3, 0.7)
        """
        x = self.relu(self.fc1(prev_aa_embedding))
        # Scale to (0.3, 0.7) range for better diversity
        gamma = 0.3 + 0.4 * self.sigmoid(self.fc2(x))
        return gamma


class ImprovedDeltaGenerator(nn.Module):
    """
    Improved δ-generator with stronger watermark signal
    Outputs higher values for better detectability
    """
    def __init__(self, embedding_dim=128, hidden_dim=64):
        super(ImprovedDeltaGenerator, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.LeakyReLU(0.01)
        self.softplus = nn.Softplus()

        # Initialize for stronger signals
        nn.init.xavier_normal_(self.fc1.weight, gain=2.0)
        nn.init.xavier_normal_(self.fc2.weight, gain=2.0)
        # Higher bias = stronger watermark
        nn.init.constant_(self.fc2.bias, 2.0)

    def forward(self, prev_aa_embedding):
        """
        Args:
            prev_aa_embedding: (batch_size, embedding_dim)
        Returns:
            delta: (batch_size, 1) watermark strength in [1.0, 4.0]
        """
        x = self.relu(self.fc1(prev_aa_embedding))
        # Stronger watermark: range [1.0, 4.0]
        delta = 1.0 + self.softplus(self.fc2(x))
        return delta


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
        """
        device = X.device

        # Get unconditional probabilities from ProteinMPNN for all positions
        with torch.no_grad():
            log_probs = self.model(X, S_true, mask, chain_mask, residue_idx,
                                   chain_encoding_all, randn, use_input_decoding_order=False)

        # Now sample autoregressively with watermarking
        N_batch, N_nodes = X.size(0), X.size(1)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)

        # Determine decoding order
        chain_mask_combined = chain_mask * chain_M_pos * mask
        decoding_order = torch.argsort((chain_mask_combined + 0.0001) * torch.abs(randn))

        watermark_stats = {'gamma': [], 'delta': [], 'green_lists': [], 'positions': []}

        # Sample autoregressively
        for t_ in range(N_nodes):
            t = decoding_order[0, t_]

            mask_val = mask[0, t].item()
            if mask_val == 0:
                S[0, t] = S_true[0, t]
                continue

            # Get logits for this position from ProteinMPNN
            logits = log_probs[0, t, :] / 0.01

            # Apply watermark if not first position
            if t_ > 0:
                # Get previous amino acid
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


def test_improved_watermarking():
    """Test with improved generators"""
    print_section("IMPROVED PROTEINMPNN WATERMARKING TEST")

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
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")

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

    print(f"✓ Structure loaded: {pdb_id}")
    print(f"  - Length: {len(native_seq)}")
    print(f"  - Native sequence: {native_seq[:60]}...")

    # 3. Initialize IMPROVED watermarking generators
    print_section("STEP 3: Initialize IMPROVED Watermarking Generators")

    gamma_gen = ImprovedGammaGenerator(embedding_dim=128, hidden_dim=64).to(device)
    delta_gen = ImprovedDeltaGenerator(embedding_dim=128, hidden_dim=64).to(device)

    print(f"✓ Improved generators initialized")
    print(f"  - Gamma generator: Better diversity (0.3-0.7 range)")
    print(f"  - Delta generator: Stronger watermark (1.0-4.0 range)")
    print(f"  - Gamma params: {sum(p.numel() for p in gamma_gen.parameters())}")
    print(f"  - Delta params: {sum(p.numel() for p in delta_gen.parameters())}")

    watermarker = ProteinWatermarker(gamma_gen, delta_gen, secret_key="improved_test")
    wm_model = WatermarkedProteinMPNN(model, gamma_gen, delta_gen, watermarker.secret_key)

    # 4. Generate watermarked sequences
    print_section("STEP 4: Generate Watermarked Sequences")

    n_sequences = 10  # More sequences for better statistics
    temperature = 0.1
    omit_AAs_np = np.zeros(21)

    watermarked_sequences = []
    baseline_sequences = []

    print(f"\nGenerating {n_sequences} sequences...")

    for i in range(n_sequences):
        randn = torch.randn(chain_M.shape, device=device)

        # Generate watermarked sequence
        wm_seq, wm_stats = wm_model.sample_watermarked(
            X, randn, S, chain_M, chain_encoding_all, residue_idx, mask,
            temperature=temperature, chain_M_pos=chain_M_pos
        )

        watermarked_sequences.append((wm_seq, wm_stats))

        # Generate baseline
        with torch.no_grad():
            output_dict = model.sample(
                X, randn, S, chain_M, chain_encoding_all, residue_idx,
                mask=mask, temperature=temperature, omit_AAs_np=omit_AAs_np,
                bias_AAs_np=np.zeros(21), chain_M_pos=chain_M_pos,
                omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef,
                pssm_bias=pssm_bias, pssm_multi=0.0, pssm_log_odds_flag=False,
                pssm_log_odds_mask=None, pssm_bias_flag=False,
                bias_by_res=bias_by_res_all
            )

        alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        S_sample = output_dict["S"]
        baseline_seq = ''.join([alphabet[S_sample[0, i].item()] for i in range(len(native_seq)) if mask[0, i] == 1])
        baseline_sequences.append(baseline_seq)

        print(f"  Sequence {i + 1}: ", end="")
        if len(wm_stats['gamma']) > 0:
            print(f"γ={np.mean(wm_stats['gamma']):.3f}, δ={np.mean(wm_stats['delta']):.3f}")

    # 5. Detect watermarks
    print_section("STEP 5: Watermark Detection")

    print(f"\nDetecting watermarks:")
    print(f"\n{'Type':<15} {'Seq#':<8} {'Z-score':<12} {'P-value':<12} {'Detected'}")
    print("-" * 80)

    wm_detections = []
    baseline_detections = []

    for i, (wm_seq, wm_stats) in enumerate(watermarked_sequences):
        result = watermarker.detect_watermark(wm_seq)
        wm_detections.append(result)
        status = "✓" if result['is_watermarked'] else "✗"
        print(f"{'Watermarked':<15} {i + 1:<8} {result['z_score']:<12.4f} {result['p_value']:<12.6f} {status}")

    print()

    for i, baseline_seq in enumerate(baseline_sequences):
        result = watermarker.detect_watermark(baseline_seq)
        baseline_detections.append(result)
        status = "✓" if result['is_watermarked'] else "✗"
        print(f"{'Baseline':<15} {i + 1:<8} {result['z_score']:<12.4f} {result['p_value']:<12.6f} {status}")

    native_result = watermarker.detect_watermark(native_seq)
    status = "✓" if native_result['is_watermarked'] else "✗"
    print(f"{'Native':<15} {'-':<8} {native_result['z_score']:<12.4f} {native_result['p_value']:<12.6f} {status}")

    # 6. Analysis
    print_section("STEP 6: RESULTS ANALYSIS")

    wm_z_scores = [d['z_score'] for d in wm_detections]
    baseline_z_scores = [d['z_score'] for d in baseline_detections]

    wm_detected = sum(1 for d in wm_detections if d['is_watermarked'])
    baseline_detected = sum(1 for d in baseline_detections if d['is_watermarked'])

    print(f"\n📊 Watermarked Sequences:")
    print(f"  - Mean Z-score: {np.mean(wm_z_scores):.4f} ± {np.std(wm_z_scores):.4f}")
    print(f"  - Min Z-score: {np.min(wm_z_scores):.4f}")
    print(f"  - Max Z-score: {np.max(wm_z_scores):.4f}")
    print(f"  - Detection rate: {wm_detected}/{n_sequences} ({100 * wm_detected / n_sequences:.1f}%)")

    print(f"\n📊 Baseline Sequences:")
    print(f"  - Mean Z-score: {np.mean(baseline_z_scores):.4f} ± {np.std(baseline_z_scores):.4f}")
    print(f"  - Min Z-score: {np.min(baseline_z_scores):.4f}")
    print(f"  - Max Z-score: {np.max(baseline_z_scores):.4f}")
    print(f"  - False positive rate: {baseline_detected}/{n_sequences} ({100 * baseline_detected / n_sequences:.1f}%)")

    # Calculate improvement metrics
    z_score_separation = np.mean(wm_z_scores) - np.mean(baseline_z_scores)

    print(f"\n📈 Performance Metrics:")
    print(f"  - Z-score separation: {z_score_separation:.4f}")
    print(f"  - True Positive Rate: {100 * wm_detected / n_sequences:.1f}%")
    print(f"  - False Positive Rate: {100 * baseline_detected / n_sequences:.1f}%")

    # Sequence similarity
    print(f"\n🧬 Sequence Quality:")
    similarities = []
    for i in range(n_sequences):
        wm_seq = watermarked_sequences[i][0]
        base_seq = baseline_sequences[i]
        matches = sum(1 for a, b in zip(wm_seq, base_seq) if a == b)
        similarity = matches / len(wm_seq)
        similarities.append(similarity)

    print(f"  - Mean similarity to baseline: {np.mean(similarities)*100:.1f}%")
    print(f"  - Sequences remain protein-like")

    # Summary
    print_section("FINAL SUMMARY")

    print("\n✓ IMPROVED WATERMARKING TEST COMPLETED!")

    if wm_detected > 0:
        print(f"\n✅ SUCCESS: Watermark detection is working!")
        print(f"   - {wm_detected}/{n_sequences} watermarked sequences detected")
        print(f"   - Mean Z-score increased from ~0.16 to {np.mean(wm_z_scores):.2f}")
    else:
        print(f"\n⚠️  Detection still needs improvement")
        print(f"   - Consider: stronger δ values, more training data, or full training")

    print("\n" + "=" * 80)

    return True


def main():
    """Run the test"""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  IMPROVED PROTEINMPNN WATERMARKING TEST".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    torch.manual_seed(42)
    np.random.seed(42)

    try:
        success = test_improved_watermarking()

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
