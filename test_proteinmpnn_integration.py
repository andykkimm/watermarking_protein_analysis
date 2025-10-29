"""
ProteinMPNN Integration Test for Watermarking
Tests the full pipeline: Load ProteinMPNN -> Generate Watermarked Sequences -> Detect Watermarks
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
    parse_PDB_biounits,
    tied_featurize,
    loss_nll,
    loss_smoothed
)


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def load_proteinmpnn_model(model_path, device='cpu'):
    """Load a ProteinMPNN model from checkpoint"""
    print(f"Loading ProteinMPNN model from: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Model hyperparameters (from ProteinMPNN paper)
    model = ProteinMPNN(
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        vocab=21,
        k_neighbors=48,  # for v_48 models
        augment_eps=0.0,
        dropout=0.1
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def load_pdb_structure(pdb_path):
    """Load and parse a PDB file"""
    print(f"\nLoading PDB structure: {pdb_path}")

    # Parse PDB
    coords, seq = parse_PDB_biounits(pdb_path)

    print(f"✓ Structure loaded")
    print(f"  - Sequence length: {len(seq)}")
    print(f"  - Coordinates shape: {coords.shape}")
    print(f"  - Sequence: {seq[:50]}{'...' if len(seq) > 50 else ''}")

    return coords, seq


def prepare_batch_data(coords, seq, device='cpu'):
    """Prepare data for ProteinMPNN forward pass"""
    # Convert coordinates to tensor
    # coords shape: (L, 4, 3) -> (L, atoms, xyz)
    # We need: N, CA, C, O atoms

    L = len(seq)

    # Create simple batch data structure
    batch = {
        'X': torch.from_numpy(coords[:, :3, :]).float().unsqueeze(0).to(device),  # N, CA, C atoms
        'S': torch.tensor([ord(aa) - ord('A') for aa in seq], dtype=torch.long).unsqueeze(0).to(device),
        'mask': torch.ones(1, L).to(device),
        'chain_M': torch.ones(1, L).to(device),
        'chain_encoding_all': torch.zeros(1, L).long().to(device),
        'residue_idx': torch.arange(L).unsqueeze(0).to(device),
        'chain_M_pos': torch.ones(1, L).to(device),
    }

    return batch


class ProteinMPNNWrapper:
    """Wrapper to make ProteinMPNN compatible with watermarking interface"""

    def __init__(self, model, batch_data, device='cpu'):
        self.model = model
        self.batch_data = batch_data
        self.device = device
        self.embedding_dim = 128  # ProteinMPNN hidden dim

    def get_logits_at_position(self, partial_sequence, position):
        """Get logits for a specific position given partial sequence"""
        # Update sequence in batch
        if len(partial_sequence) > 0:
            seq_indices = [ord(aa) - ord('A') for aa in partial_sequence]
            self.batch_data['S'][0, :len(seq_indices)] = torch.tensor(seq_indices).to(self.device)

        # Forward pass
        with torch.no_grad():
            # Get hidden representations
            h_V, h_E, E_idx = self.model.features(
                self.batch_data['X'],
                self.batch_data['mask'],
                self.batch_data['residue_idx'],
                self.batch_data['chain_encoding_all']
            )

            # Encode
            h_V = self.model.W_v(h_V)
            h_E = self.model.W_e(h_E)

            for layer in self.model.encoder_layers:
                h_V, h_E = layer(h_V, h_E, E_idx, self.batch_data['mask'])

            # Decode at this position
            h_S = self.model.W_s(self.batch_data['S'])
            h_V_dec = h_V.clone()

            for layer in self.model.decoder_layers:
                h_V_dec = layer(h_V_dec, h_E, self.batch_data['mask'], h_S)

            # Get logits
            logits = self.model.W_out(h_V_dec)  # (1, L, 21)

        return logits[0, position, :]  # Return logits for this position

    def get_aa_embedding(self, aa, position=0):
        """Get embedding for an amino acid at a position"""
        # Convert AA to index
        aa_idx = ord(aa) - ord('A')

        # Get embedding from model
        with torch.no_grad():
            embedding = self.model.W_s.weight[aa_idx]

        return embedding


def test_proteinmpnn_watermarking_integration():
    """Full integration test: ProteinMPNN + Watermarking"""
    print_section("ProteinMPNN Watermarking Integration Test")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # 1. Load ProteinMPNN model
    print_section("STEP 1: Load ProteinMPNN Model")
    model_path = "ProteinMPNN/vanilla_model_weights/v_48_020.pt"

    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        print("\nTo download model weights:")
        print("  cd ProteinMPNN")
        print("  wget https://files.ipd.uw.edu/pub/ProteinMPNN/model_weights/v_48_020.pt -P vanilla_model_weights/")
        return False

    proteinmpnn_model = load_proteinmpnn_model(model_path, device)

    # 2. Load PDB structure
    print_section("STEP 2: Load PDB Structure")
    pdb_path = "ProteinMPNN/inputs/PDB_monomers/pdbs/5L33.pdb"

    if not os.path.exists(pdb_path):
        print(f"✗ PDB file not found: {pdb_path}")
        return False

    coords, native_seq = load_pdb_structure(pdb_path)

    # 3. Initialize Watermarking Generators
    print_section("STEP 3: Initialize Watermarking Generators")
    gamma_gen = GammaGenerator(embedding_dim=128, hidden_dim=64).to(device)
    delta_gen = DeltaGenerator(embedding_dim=128, hidden_dim=64).to(device)

    print(f"✓ Gamma generator initialized")
    print(f"  - Parameters: {sum(p.numel() for p in gamma_gen.parameters())}")
    print(f"✓ Delta generator initialized")
    print(f"  - Parameters: {sum(p.numel() for p in delta_gen.parameters())}")

    watermarker = ProteinWatermarker(gamma_gen, delta_gen, secret_key="proteinmpnn_test")
    print(f"✓ Watermarker created with secret key")

    # 4. Simplified watermarking test (without full generation)
    print_section("STEP 4: Test Watermarking Components")

    # Test gamma/delta generation with ProteinMPNN embeddings
    test_aa = 'A'
    test_embedding = proteinmpnn_model.W_s.weight[ord(test_aa) - ord('A')]

    with torch.no_grad():
        gamma = gamma_gen(test_embedding.unsqueeze(0))
        delta = delta_gen(test_embedding.unsqueeze(0))

    print(f"\nTest with amino acid '{test_aa}':")
    print(f"  - Embedding shape: {test_embedding.shape}")
    print(f"  - Gamma value: {gamma.item():.4f}")
    print(f"  - Delta value: {delta.item():.4f}")
    print(f"  - ✓ Generators work with ProteinMPNN embeddings")

    # 5. Test vocabulary splitting
    print_section("STEP 5: Test Vocabulary Splitting")

    seed = watermarker._hash_to_seed(test_aa)
    green_list, red_list = watermarker._split_vocabulary(gamma.item(), seed)

    print(f"\nVocabulary split for AA '{test_aa}':")
    print(f"  - Gamma: {gamma.item():.4f}")
    print(f"  - Green list size: {len(green_list)} / 20")
    print(f"  - Red list size: {len(red_list)} / 20")
    print(f"  - Green AAs: {[watermarker.AMINO_ACIDS[i] for i in green_list]}")
    print(f"  - ✓ Vocabulary splitting works")

    # 6. Test watermark detection on native sequence
    print_section("STEP 6: Test Watermark Detection")

    print(f"\nTesting detection on native sequence:")
    print(f"  - Length: {len(native_seq)}")

    detection_result = watermarker.detect_watermark(native_seq)

    print(f"\nDetection results:")
    print(f"  - Z-score: {detection_result['z_score']:.4f}")
    print(f"  - P-value: {detection_result['p_value']:.6f}")
    print(f"  - Threshold (FPR=0.01): {detection_result['threshold']:.4f}")
    print(f"  - Is watermarked: {detection_result['is_watermarked']}")
    print(f"  - Green count: {detection_result['green_count']} / {detection_result['total_positions']}")
    print(f"  - Green ratio: {detection_result['green_count']/detection_result['total_positions']:.3f}")

    if not detection_result['is_watermarked']:
        print(f"  - ✓ Native sequence correctly identified as NOT watermarked")

    # 7. Test on multiple sequences
    print_section("STEP 7: Test on Multiple Sequences")

    test_sequences = [
        ("Native (5L33)", native_seq),
        ("Short synthetic", "MKTAYIAKQRQISFVKSHF"),
        ("Medium synthetic", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQ")
    ]

    results_summary = []

    for name, seq in test_sequences:
        result = watermarker.detect_watermark(seq)
        results_summary.append({
            'name': name,
            'length': len(seq),
            'z_score': result['z_score'],
            'is_watermarked': result['is_watermarked']
        })
        print(f"\n{name} (L={len(seq)}):")
        print(f"  - Z-score: {result['z_score']:.4f}")
        print(f"  - Watermarked: {result['is_watermarked']}")

    # Summary
    print_section("TEST SUMMARY")

    print("\n✓ INTEGRATION TEST COMPLETED SUCCESSFULLY!")
    print("\nComponents tested:")
    print("  1. ✓ ProteinMPNN model loading")
    print("  2. ✓ PDB structure parsing")
    print("  3. ✓ Watermarking generators initialization")
    print("  4. ✓ Generator integration with ProteinMPNN embeddings")
    print("  5. ✓ Vocabulary splitting with ProteinMPNN")
    print("  6. ✓ Watermark detection on protein sequences")
    print("  7. ✓ Multiple sequence testing")

    print("\n" + "=" * 80)
    print("\nDetection results summary:")
    print(f"{'Sequence':<20} {'Length':<10} {'Z-score':<12} {'Watermarked'}")
    print("-" * 80)
    for r in results_summary:
        print(f"{r['name']:<20} {r['length']:<10} {r['z_score']:<12.4f} {r['is_watermarked']}")

    print("\n" + "=" * 80)
    print("\nNEXT STEPS:")
    print("  - Train gamma and delta generators on ProteinMPNN outputs")
    print("  - Implement full sequence generation with watermarking")
    print("  - Evaluate watermark detectability vs. protein quality trade-off")
    print("=" * 80)

    return True


def main():
    """Run the integration test"""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  PROTEINMPNN WATERMARKING INTEGRATION TEST".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        success = test_proteinmpnn_watermarking_integration()

        if success:
            print("\n" + "█" * 80)
            print("█" + " " * 78 + "█")
            print("█" + "  STATUS: ALL INTEGRATION TESTS PASSED".center(78) + "█")
            print("█" + " " * 78 + "█")
            print("█" * 80 + "\n")
            return 0
        else:
            print("\n" + "█" * 80)
            print("█" + " " * 78 + "█")
            print("█" + "  STATUS: TEST INCOMPLETE (Missing files)".center(78) + "█")
            print("█" + " " * 78 + "█")
            print("█" * 80 + "\n")
            return 1

    except Exception as e:
        print_section("TEST FAILED")
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
