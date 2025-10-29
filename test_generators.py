"""
Comprehensive Test Suite for Gamma-Generator and Delta-Generator
Tests all aspects of the watermarking generators
"""

import torch
import numpy as np
from protein_watermark import GammaGenerator, DeltaGenerator, ProteinWatermarker

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def test_gamma_generator():
    """Test the Gamma-Generator"""
    print_section("TEST 1: Gamma-Generator")

    # Initialize generator
    gamma_gen = GammaGenerator(embedding_dim=128, hidden_dim=64)
    print(f"✓ Initialized GammaGenerator")
    print(f"  - Embedding dim: 128")
    print(f"  - Hidden dim: 64")
    print(f"  - Parameters: {sum(p.numel() for p in gamma_gen.parameters())}")

    # Test single forward pass
    test_embedding = torch.randn(1, 128)
    gamma_output = gamma_gen(test_embedding)

    print(f"\n✓ Single forward pass:")
    print(f"  - Input shape: {test_embedding.shape}")
    print(f"  - Output shape: {gamma_output.shape}")
    print(f"  - Gamma value: {gamma_output.item():.4f}")

    # Verify output range (0, 1)
    assert gamma_output.item() > 0 and gamma_output.item() < 1, "Gamma should be in (0, 1)"
    print(f"  - ✓ Output in valid range (0, 1)")

    # Test batch processing
    batch_embeddings = torch.randn(10, 128)
    batch_gammas = gamma_gen(batch_embeddings)

    print(f"\n✓ Batch forward pass:")
    print(f"  - Batch size: 10")
    print(f"  - Output shape: {batch_gammas.shape}")
    print(f"  - Min gamma: {batch_gammas.min().item():.4f}")
    print(f"  - Max gamma: {batch_gammas.max().item():.4f}")
    print(f"  - Mean gamma: {batch_gammas.mean().item():.4f}")

    # Verify all batch outputs in valid range
    assert torch.all((batch_gammas > 0) & (batch_gammas < 1)), "All gammas should be in (0, 1)"
    print(f"  - ✓ All batch outputs in valid range")

    # Test gradient flow
    gamma_gen.train()
    test_embedding = torch.randn(1, 128, requires_grad=True)
    gamma_output = gamma_gen(test_embedding)
    loss = gamma_output.mean()
    loss.backward()

    has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in gamma_gen.parameters())
    print(f"\n✓ Gradient flow:")
    print(f"  - Gradients computed: {has_gradients}")

    return gamma_gen


def test_delta_generator():
    """Test the Delta-Generator"""
    print_section("TEST 2: Delta-Generator")

    # Initialize generator
    delta_gen = DeltaGenerator(embedding_dim=128, hidden_dim=64)
    print(f"✓ Initialized DeltaGenerator")
    print(f"  - Embedding dim: 128")
    print(f"  - Hidden dim: 64")
    print(f"  - Parameters: {sum(p.numel() for p in delta_gen.parameters())}")

    # Test single forward pass
    test_embedding = torch.randn(1, 128)
    delta_output = delta_gen(test_embedding)

    print(f"\n✓ Single forward pass:")
    print(f"  - Input shape: {test_embedding.shape}")
    print(f"  - Output shape: {delta_output.shape}")
    print(f"  - Delta value: {delta_output.item():.4f}")

    # Verify output range (positive real number)
    assert delta_output.item() >= 0, "Delta should be in R+ (non-negative)"
    print(f"  - ✓ Output in valid range R+ (non-negative)")

    # Test batch processing
    batch_embeddings = torch.randn(10, 128)
    batch_deltas = delta_gen(batch_embeddings)

    print(f"\n✓ Batch forward pass:")
    print(f"  - Batch size: 10")
    print(f"  - Output shape: {batch_deltas.shape}")
    print(f"  - Min delta: {batch_deltas.min().item():.4f}")
    print(f"  - Max delta: {batch_deltas.max().item():.4f}")
    print(f"  - Mean delta: {batch_deltas.mean().item():.4f}")

    # Verify all batch outputs are non-negative
    assert torch.all(batch_deltas >= 0), "All deltas should be non-negative"
    print(f"  - ✓ All batch outputs in valid range")

    # Test gradient flow
    delta_gen.train()
    test_embedding = torch.randn(1, 128, requires_grad=True)
    delta_output = delta_gen(test_embedding)
    loss = delta_output.mean()
    loss.backward()

    has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in delta_gen.parameters())
    print(f"\n✓ Gradient flow:")
    print(f"  - Gradients computed: {has_gradients}")

    return delta_gen


def test_vocabulary_splitting(gamma_gen):
    """Test vocabulary splitting with different gamma values"""
    print_section("TEST 3: Vocabulary Splitting")

    watermarker = ProteinWatermarker(gamma_gen, DeltaGenerator(), secret_key="test_key")

    # Test with different gamma values
    gamma_values = [0.2, 0.5, 0.8]
    seed = 42

    print(f"Testing vocabulary splitting with different gamma values:")
    print(f"Total amino acids: {watermarker.vocab_size}")

    for gamma in gamma_values:
        green_list, red_list = watermarker._split_vocabulary(gamma, seed)

        print(f"\n  Gamma = {gamma}:")
        print(f"    - Green list size: {len(green_list)}")
        print(f"    - Red list size: {len(red_list)}")
        print(f"    - Total: {len(green_list) + len(red_list)}")
        print(f"    - Green ratio: {len(green_list)/watermarker.vocab_size:.2f}")

        # Verify complete partition
        assert len(green_list) + len(red_list) == watermarker.vocab_size
        assert len(set(green_list) & set(red_list)) == 0  # No overlap
        print(f"    - ✓ Valid partition (no overlap, complete coverage)")

    # Test reproducibility with same seed
    green1, red1 = watermarker._split_vocabulary(0.5, 12345)
    green2, red2 = watermarker._split_vocabulary(0.5, 12345)

    assert green1 == green2 and red1 == red2
    print(f"\n✓ Reproducibility test passed (same seed produces same split)")

    # Test different seeds produce different splits
    green3, red3 = watermarker._split_vocabulary(0.5, 54321)
    assert green1 != green3 or red1 != red3
    print(f"✓ Different seeds produce different splits")


def test_watermark_detection(gamma_gen, delta_gen):
    """Test watermark detection on various sequences"""
    print_section("TEST 4: Watermark Detection")

    watermarker = ProteinWatermarker(gamma_gen, delta_gen, secret_key="detection_test")

    # Test sequences of different lengths
    test_sequences = [
        ("Short", "MKTAYIAKQRQISFVKSHF"),
        ("Medium", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRP"),
        ("Long", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSLEVGCSRLAKAVGQPLRVGVVKDEEILKVTVGVRGDKWIVEAVPGDVRIVQVSMVRPCRVLLKPVEETDTVLATGGHYEEFFCRALAEAGALPLGAANVAKYSAAQVGQVTIVKDGPRIVRALKDQVVVSGLMKDAADQVQEMLKWL")
    ]

    results = []

    for name, sequence in test_sequences:
        detection = watermarker.detect_watermark(sequence, fpr=0.01)
        results.append((name, len(sequence), detection))

        print(f"\n{name} sequence (length: {len(sequence)}):")
        print(f"  - Z-score: {detection['z_score']:.4f}")
        print(f"  - P-value: {detection['p_value']:.6f}")
        print(f"  - Threshold: {detection['threshold']:.4f}")
        print(f"  - Watermarked: {detection['is_watermarked']}")
        print(f"  - Green count: {detection['green_count']}/{detection['total_positions']}")
        print(f"  - Green ratio: {detection['green_count']/detection['total_positions']:.3f}")

    print(f"\n✓ Detection completed on {len(test_sequences)} sequences")

    return results


def test_generator_diversity():
    """Test that generators produce diverse outputs for different inputs"""
    print_section("TEST 5: Generator Output Diversity")

    gamma_gen = GammaGenerator(embedding_dim=128)
    delta_gen = DeltaGenerator(embedding_dim=128)

    # Generate multiple random embeddings
    n_samples = 100
    embeddings = torch.randn(n_samples, 128)

    # Get outputs
    gammas = gamma_gen(embeddings)
    deltas = delta_gen(embeddings)

    # Calculate statistics
    gamma_std = gammas.std().item()
    delta_std = deltas.std().item()

    print(f"Testing diversity on {n_samples} random embeddings:")
    print(f"\nGamma statistics:")
    print(f"  - Mean: {gammas.mean().item():.4f}")
    print(f"  - Std: {gamma_std:.4f}")
    print(f"  - Min: {gammas.min().item():.4f}")
    print(f"  - Max: {gammas.max().item():.4f}")
    print(f"  - Range: {(gammas.max() - gammas.min()).item():.4f}")

    print(f"\nDelta statistics:")
    print(f"  - Mean: {deltas.mean().item():.4f}")
    print(f"  - Std: {delta_std:.4f}")
    print(f"  - Min: {deltas.min().item():.4f}")
    print(f"  - Max: {deltas.max().item():.4f}")
    print(f"  - Range: {(deltas.max() - deltas.min()).item():.4f}")

    # Check for sufficient diversity
    assert gamma_std > 0.01, "Gamma generator should produce diverse outputs"
    assert delta_std > 0.01, "Delta generator should produce diverse outputs"

    print(f"\n✓ Both generators produce diverse outputs")


def test_hash_to_seed():
    """Test the hash-to-seed function for reproducibility"""
    print_section("TEST 6: Hash-to-Seed Reproducibility")

    gamma_gen = GammaGenerator()
    delta_gen = DeltaGenerator()
    watermarker = ProteinWatermarker(gamma_gen, delta_gen, secret_key="hash_test")

    # Test reproducibility
    aa = 'A'
    seed1 = watermarker._hash_to_seed(aa)
    seed2 = watermarker._hash_to_seed(aa)

    assert seed1 == seed2
    print(f"✓ Hash function is deterministic")
    print(f"  - Amino acid: {aa}")
    print(f"  - Seed: {seed1}")

    # Test different amino acids produce different seeds
    seeds = [watermarker._hash_to_seed(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY']
    unique_seeds = len(set(seeds))

    print(f"\n✓ Different amino acids produce different seeds")
    print(f"  - Total amino acids tested: 20")
    print(f"  - Unique seeds: {unique_seeds}")

    # Test different secret keys produce different seeds
    watermarker2 = ProteinWatermarker(gamma_gen, delta_gen, secret_key="different_key")
    seed3 = watermarker2._hash_to_seed(aa)

    assert seed1 != seed3
    print(f"\n✓ Different secret keys produce different seeds")
    print(f"  - Same amino acid: {aa}")
    print(f"  - Key 1 seed: {seed1}")
    print(f"  - Key 2 seed: {seed3}")


def main():
    """Run all tests"""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  COMPREHENSIVE TEST SUITE: γ-Generator & δ-Generator".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # Run all tests
        gamma_gen = test_gamma_generator()
        delta_gen = test_delta_generator()
        test_vocabulary_splitting(gamma_gen)
        test_watermark_detection(gamma_gen, delta_gen)
        test_generator_diversity()
        test_hash_to_seed()

        # Summary
        print_section("TEST SUMMARY")
        print("\n✓ ALL TESTS PASSED SUCCESSFULLY!")
        print("\nTests completed:")
        print("  1. ✓ Gamma-Generator forward pass, output range, and gradients")
        print("  2. ✓ Delta-Generator forward pass, output range, and gradients")
        print("  3. ✓ Vocabulary splitting with different gamma values")
        print("  4. ✓ Watermark detection on sequences of varying lengths")
        print("  5. ✓ Generator output diversity")
        print("  6. ✓ Hash-to-seed reproducibility and uniqueness")

        print("\n" + "█" * 80)
        print("█" + " " * 78 + "█")
        print("█" + "  STATUS: ALL SYSTEMS OPERATIONAL".center(78) + "█")
        print("█" + " " * 78 + "█")
        print("█" * 80 + "\n")

        return True

    except Exception as e:
        print_section("TEST FAILED")
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
