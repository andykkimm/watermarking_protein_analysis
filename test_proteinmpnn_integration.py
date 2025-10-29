"""
Real ProteinMPNN Integration Test with Watermark Generation
This test actually loads ProteinMPNN and generates watermarked protein sequences
"""

import torch
import numpy as np
import sys
import os
import json
from pathlib import Path

# Add ProteinMPNN to path
sys.path.insert(0, str(Path(__file__).parent / "ProteinMPNN"))

from protein_watermark import GammaGenerator, DeltaGenerator, ProteinWatermarker
from ProteinMPNN.protein_mpnn_utils import (
    ProteinMPNN,
    StructureDatasetPDB,
    tied_featurize,
    parse_PDB,
    gather_nodes,
    cat_neighbors_nodes
)
import torch.nn.functional as F


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def generate_watermarked_sequence_proteinmpnn(
    model,
    X,
    randn,
    S_true,
    chain_mask,
    chain_encoding_all,
    residue_idx,
    mask,
    gamma_gen,
    delta_gen,
    secret_key,
    temperature=0.1,
    omit_AAs_np=None,
    bias_AAs_np=None,
    chain_M_pos=None
):
    """
    Modified ProteinMPNN sampling with watermark embedding
    Based on model.sample() but adds watermarking logic
    """
    device = X.device
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

    # Prepare node and edge embeddings (same as ProteinMPNN)
    E, E_idx = model.features(X, mask, residue_idx, chain_encoding_all)
    h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
    h_E = model.W_e(E)

    # Encoder is unmasked self-attention
    mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
    mask_attend = mask.unsqueeze(-1) * mask_attend
    for layer in model.encoder_layers:
        h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

    # Decoder setup
    chain_mask = chain_mask * chain_M_pos * mask
    decoding_order = torch.argsort((chain_mask + 0.0001) * (torch.abs(randn)))

    N_batch, N_nodes = X.size(0), X.size(1)
    S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
    h_S = torch.zeros_like(h_V, device=device)

    watermarker = ProteinWatermarker(gamma_gen, delta_gen, secret_key)
    watermark_stats = {'gamma': [], 'delta': [], 'green_lists': [], 'positions': []}

    # Build masks for autoregressive decoding
    mask_size = E_idx.shape[1]
    permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
    order_mask_backward = torch.einsum('ij, biq, bjp->bqp',
                                       (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
                                       permutation_matrix_reverse,
                                       permutation_matrix_reverse)
    mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
    mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
    mask_bw = mask_1D * mask_attend
    mask_fw = mask_1D * (1. - mask_attend)

    h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
    h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
    h_EXV_encoder_fw = mask_fw * h_EXV_encoder

    # Autoregressive sampling with watermarking
    for t_ in range(N_nodes):
        t = decoding_order[:, t_]

        # Get mask for this position
        chain_mask_gathered = torch.gather(chain_mask, 1, t[:, None])
        mask_gathered = torch.gather(mask, 1, t[:, None])

        if (mask_gathered == 0).all():
            # Padded or missing regions
            S_t = torch.gather(S_true, 1, t[:, None])
        else:
            # Decoder layers
            E_idx_t = torch.gather(E_idx, 1, t[:, None, None].repeat(1, 1, E_idx.shape[-1]))
            h_E_t = torch.gather(h_E, 1, t[:, None, None, None].repeat(1, 1, h_E.shape[-2], h_E.shape[-1]))

            h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
            h_EXV_encoder_t = torch.gather(h_EXV_encoder_fw, 1,
                                          t[:, None, None, None].repeat(1, 1, h_EXV_encoder_fw.shape[-2],
                                                                        h_EXV_encoder_fw.shape[-1]))

            mask_t = torch.gather(mask, 1, t[:, None])

            for l_idx, layer in enumerate(model.decoder_layers):
                h_ESV_decoder_t = torch.gather(h_V, 1, t[:, None, None].repeat(1, 1, h_V.shape[-1]))
                h_V_t = torch.gather(h_V, 1, t[:, None, None].repeat(1, 1, h_V.shape[-1]))
                h_ESV_t = cat_neighbors_nodes(h_V_t, h_ES_t, E_idx_t)
                h_ESV_t = h_ESV_t + h_EXV_encoder_t
                h_V_t = layer(h_V_t, h_ESV_t, mask_t)

                # Scatter back
                h_V = h_V.scatter(1, t[:, None, None].repeat(1, 1, h_V.shape[-1]), h_V_t)

            # Get logits
            logits = model.W_out(h_V_t)
            logits = logits[:, 0, :]  # [B, 21]

            # Apply watermark if not first position
            if t_ > 0:
                # Get previous amino acid
                prev_idx = torch.gather(S, 1, decoding_order[:, t_ - 1:t_])
                prev_aa = alphabet[prev_idx[0, 0].item()]

                # Get embedding of previous AA
                prev_emb = model.W_s.weight[prev_idx[0, 0]]

                # Generate gamma and delta
                with torch.no_grad():
                    gamma = gamma_gen(prev_emb.unsqueeze(0)).item()
                    delta = delta_gen(prev_emb.unsqueeze(0)).item()

                # Split vocabulary
                seed = watermarker._hash_to_seed(prev_aa)
                green_list, red_list = watermarker._split_vocabulary(gamma, seed)

                # Apply watermark: add delta to green amino acids
                for aa_idx in green_list:
                    logits[:, aa_idx] += delta

                # Store watermark stats
                watermark_stats['gamma'].append(gamma)
                watermark_stats['delta'].append(delta)
                watermark_stats['green_lists'].append(green_list)
                watermark_stats['positions'].append(t_.item())

            # Sample from modified logits
            probs = F.softmax(logits / temperature, dim=-1)

            if omit_AAs_np is not None:
                probs_masked = probs * (1.0 - torch.tensor(omit_AAs_np, device=device))
                probs_masked = probs_masked / probs_masked.sum(dim=-1, keepdim=True)
                S_t = torch.multinomial(probs_masked, 1)
            else:
                S_t = torch.multinomial(probs, 1)

            # Update sequence embedding
            h_S_t = model.W_s(S_t)
            h_S = h_S.scatter(1, t[:, None, None].repeat(1, 1, h_S.shape[-1]), h_S_t)

        # Store sampled amino acid
        S = S.scatter(1, t[:, None], S_t)

    # Convert to sequence string
    sequence = ''.join([alphabet[S[0, i].item()] for i in range(N_nodes) if mask[0, i] == 1])

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
        wm_seq, wm_stats = generate_watermarked_sequence_proteinmpnn(
            model, X, randn, S, chain_M, chain_encoding_all, residue_idx, mask,
            gamma_gen, delta_gen, watermarker.secret_key,
            temperature=temperature,
            omit_AAs_np=omit_AAs_np,
            chain_M_pos=chain_M_pos
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
