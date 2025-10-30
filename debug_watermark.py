"""
Debug script to understand why watermarking isn't working
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "ProteinMPNN"))

from ProteinMPNN.protein_mpnn_utils import ProteinMPNN, parse_PDB, StructureDatasetPDB, tied_featurize

# Simple test
torch.manual_seed(42)
np.random.seed(42)

device = 'cpu'

# Load model
checkpoint = torch.load("ProteinMPNN/vanilla_model_weights/v_48_020.pt", map_location=device)
model = ProteinMPNN(num_letters=21, node_features=128, edge_features=128, hidden_dim=128,
                    num_encoder_layers=3, num_decoder_layers=3, vocab=21,
                    k_neighbors=48, augment_eps=0.0, dropout=0.1)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Load structure
pdb_dict_list = parse_PDB("ProteinMPNN/inputs/PDB_monomers/pdbs/5L33.pdb")
dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=10000)
structure_dict = dataset_valid[0]
batch = [structure_dict]

X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, \
    visible_list_list, masked_list_list, masked_chain_length_list_list, \
    chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, \
    pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(
    batch, device, None, None, None, None, None, None, ca_only=False)

print("Testing watermark application...")
print(f"Sequence length: {lengths[0]}")

# Let's manually check what happens when we add a strong bias
randn = torch.randn(chain_M.shape, device=device)

# Generate baseline
output_baseline = model.sample(
    X, randn, S, chain_M, chain_encoding_all, residue_idx,
    mask=mask, temperature=0.1, omit_AAs_np=np.zeros(21),
    bias_AAs_np=np.zeros(21), chain_M_pos=chain_M_pos,
    omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef,
    pssm_bias=pssm_bias, pssm_multi=0.0, pssm_log_odds_flag=False,
    pssm_log_odds_mask=None, pssm_bias_flag=False,
    bias_by_res=bias_by_res_all
)

S_baseline = output_baseline["S"]
alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
baseline_seq = ''.join([alphabet[S_baseline[0, i].item()] for i in range(lengths[0])])

print(f"\nBaseline sequence: {baseline_seq[:60]}...")

# Now let's test if adding a STRONG global bias actually changes anything
# Let's bias towards Alanine (A = index 0)
strong_bias = np.zeros(21)
strong_bias[0] = 10.0  # Very strong bias towards A

output_biased = model.sample(
    X, randn, S, chain_M, chain_encoding_all, residue_idx,
    mask=mask, temperature=0.1, omit_AAs_np=np.zeros(21),
    bias_AAs_np=strong_bias, chain_M_pos=chain_M_pos,
    omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef,
    pssm_bias=pssm_bias, pssm_multi=0.0, pssm_log_odds_flag=False,
    pssm_log_odds_mask=None, pssm_bias_flag=False,
    bias_by_res=bias_by_res_all
)

S_biased = output_biased["S"]
biased_seq = ''.join([alphabet[S_biased[0, i].item()] for i in range(lengths[0])])

print(f"Biased sequence:   {biased_seq[:60]}...")

# Count alanines
baseline_A_count = baseline_seq.count('A')
biased_A_count = biased_seq.count('A')

print(f"\nAlanine (A) count:")
print(f"  Baseline: {baseline_A_count}/{lengths[0]} ({100*baseline_A_count/lengths[0]:.1f}%)")
print(f"  Biased:   {biased_A_count}/{lengths[0]} ({100*biased_A_count/lengths[0]:.1f}%)")

if biased_A_count > baseline_A_count:
    print(f"\n✓ Bias is working! Increase of {biased_A_count - baseline_A_count} alanines")
else:
    print(f"\n✗ Bias may not be working as expected")

print("\n" + "="*80)
print("This test shows whether ProteinMPNN's bias_AAs_np parameter actually works")
