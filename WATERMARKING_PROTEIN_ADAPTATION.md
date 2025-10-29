# Adapting Token-Specific Watermarking to Protein Generation

## Overview

This document outlines how to adapt the token-specific watermarking method from the paper to **protein sequence generation using ProteinMPNN**.

## Key Differences: LLM Text vs Protein Generation

| Aspect | LLM Watermarking | Protein Watermarking |
|--------|------------------|---------------------|
| **Vocabulary** | ~50,000 tokens | 20 amino acids (A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y) |
| **Input** | Text prompt | Protein backbone structure (3D coordinates) |
| **Context** | Previous tokens in text | Previous amino acids + structure constraints |
| **Semantic Loss** | SimCSE (RoBERTa embeddings) | ESM/ProtBERT embeddings OR structural compatibility |
| **Generation Goal** | Coherent text | Functional, foldable protein |
| **Detectability Metric** | Z-score (green token count) | Same - Z-score with amino acid green list |

## Core Algorithm Adaptation

### 1. **Watermark Embedding (Generation)**

**For each position i in the protein sequence:**

```python
# Input: Previous amino acid embedding from ProteinMPNN
prev_aa_embedding = proteinmpnn_embedding(sequence[i-1])

# Generate token-specific parameters
γ_i = gamma_generator(prev_aa_embedding)  # Splitting ratio ∈ (0,1)
δ_i = delta_generator(prev_aa_embedding)  # Watermark logit ∈ ℝ+

# Split amino acid vocabulary into green/red lists
# Use hash of previous amino acid + secret key as random seed
seed = hash(sequence[i-1] + secret_key)
green_list, red_list = split_vocabulary(20_amino_acids, γ_i, seed)

# Get ProteinMPNN logits for position i
logits_i = proteinmpnn.get_logits(structure, sequence[:i])

# Add watermark bias to green amino acids
for aa in green_list:
    logits_i[aa] += δ_i

# Sample next amino acid from modified distribution
sequence[i] = sample(softmax(logits_i))
```

### 2. **Watermark Detection**

```python
def detect_watermark(protein_sequence, secret_key, gamma_generator):
    """
    Detect if a protein sequence contains a watermark.

    Returns:
        z_score: Higher values indicate watermarked sequence
        p_value: Statistical significance
    """
    T = len(protein_sequence)
    green_count = 0
    sum_gamma = 0
    sum_variance = 0

    for i in range(1, T):
        # Recreate green list using same seed
        seed = hash(protein_sequence[i-1] + secret_key)

        # Get gamma for this position
        prev_aa_emb = get_embedding(protein_sequence[i-1])
        gamma_i = gamma_generator(prev_aa_emb)

        # Check if current amino acid is in green list
        green_list = generate_green_list(gamma_i, seed)
        if protein_sequence[i] in green_list:
            green_count += 1

        # Accumulate statistics for z-score
        sum_gamma += gamma_i
        sum_variance += gamma_i * (1 - gamma_i)

    # Calculate z-score (Equation 3 from paper)
    z_score = (green_count - sum_gamma) / sqrt(sum_variance)
    p_value = 1 - norm.cdf(z_score)  # One-sided test

    return z_score, p_value
```

### 3. **Training Objectives**

**Multi-Objective Optimization (MOO):**

#### **Loss 1: Detection Loss** (maximize detectability)

```python
def detection_loss(gamma_values, delta_values, generated_sequences):
    """
    Maximize z-score for detectability.
    Uses differentiable approximation from Equation 4.
    """
    T = len(generated_sequences[0])

    # Differentiable approximation of green token count
    green_prob_sum = 0
    for t in range(T):
        # p_gr^(t) = probability of sampling green amino acid at position t
        # This is computed from the modified logits
        green_prob_sum += compute_green_probability(t, gamma_values[t], delta_values[t])

    # Differentiable z-score (Equation 4)
    numerator = green_prob_sum - sum(gamma_values)
    denominator = sqrt(sum([γ * (1 - γ) for γ in gamma_values]))
    z_score_hat = numerator / denominator

    # Loss: negative z-score (we want to maximize z-score)
    L_detection = -z_score_hat

    return L_detection
```

#### **Loss 2: Protein Functionality Loss** (maintain protein quality)

**Option A: Using Protein Language Model Embeddings**

```python
def semantic_loss_plm(watermarked_seq, original_seq, protein_lm='esm2'):
    """
    Use protein language model (e.g., ESM-2) embeddings.
    Similar to SimCSE in the paper, but for proteins.
    """
    # Get embeddings from pre-trained protein language model
    emb_watermarked = esm_model.encode(watermarked_seq)
    emb_original = esm_model.encode(original_seq)

    # Cosine similarity (want this close to 1)
    similarity = cosine_similarity(emb_watermarked, emb_original)

    # Loss: negative similarity
    L_semantic = -similarity

    return L_semantic
```

**Option B: Using Structural Compatibility**

```python
def semantic_loss_structure(watermarked_seq, structure, structure_predictor):
    """
    Ensure watermarked sequence still folds to similar structure.
    Uses AlphaFold or similar to predict structure.
    """
    # Predict structure from watermarked sequence
    predicted_structure = structure_predictor(watermarked_seq)

    # Compute RMSD or TM-score between original and predicted structure
    rmsd = compute_rmsd(structure, predicted_structure)

    # Loss: structural deviation
    L_semantic = rmsd

    return L_semantic
```

### 4. **Multi-Objective Optimization (MGDA)**

```python
def train_watermark_generators(proteinmpnn, train_structures, epochs=100):
    """
    Train γ-generator and δ-generator using MGDA.
    """
    # Initialize lightweight MLPs
    gamma_generator = MLP(input_dim=embedding_dim, output_dim=1, activation='sigmoid')
    delta_generator = MLP(input_dim=embedding_dim, output_dim=1, activation='relu')

    optimizer = torch.optim.Adam(
        list(gamma_generator.parameters()) + list(delta_generator.parameters()),
        lr=1e-4
    )

    for epoch in range(epochs):
        for structure in train_structures:
            # Generate original sequence (no watermark)
            original_seq = proteinmpnn.generate(structure)

            # Generate watermarked sequence
            watermarked_seq, gamma_vals, delta_vals = generate_watermarked(
                proteinmpnn, structure, gamma_generator, delta_generator
            )

            # Compute two losses
            L_detection = detection_loss(gamma_vals, delta_vals, watermarked_seq)
            L_semantic = semantic_loss_plm(watermarked_seq, original_seq)

            # Compute gradients
            grad_detection = torch.autograd.grad(L_detection,
                                                [gamma_generator.parameters(),
                                                 delta_generator.parameters()],
                                                retain_graph=True)
            grad_semantic = torch.autograd.grad(L_semantic,
                                               [gamma_generator.parameters(),
                                                delta_generator.parameters()])

            # MGDA: find optimal gradient direction (Appendix C)
            lambda_star = compute_mgda_weight(grad_detection, grad_semantic)
            combined_grad = lambda_star * grad_detection + (1 - lambda_star) * grad_semantic

            # Update parameters
            optimizer.zero_grad()
            set_gradients(gamma_generator, delta_generator, combined_grad)
            optimizer.step()
```

## Implementation Steps

### Step 1: Setup ProteinMPNN

```bash
# Clone ProteinMPNN repository
git clone https://github.com/dauparas/ProteinMPNN.git
cd ProteinMPNN

# Install dependencies
conda create -n proteinmpnn_watermark python=3.9
conda activate proteinmpnn_watermark
pip install torch biopython numpy
```

### Step 2: Implement Generator Networks

The γ-generator and δ-generator are lightweight MLPs that take the embedding of the previous amino acid.

**Key Design Choices:**
- **Input**: ProteinMPNN's amino acid embeddings (typically 128-dim)
- **Architecture**: 2-layer MLP with hidden dimension 64
- **Output**:
  - γ-generator: Sigmoid activation → (0, 1)
  - δ-generator: ReLU or Softplus → ℝ+

### Step 3: Integrate with ProteinMPNN Generation

Modify ProteinMPNN's sampling function to:
1. Get logits for next amino acid
2. Apply watermark bias based on γ and δ
3. Sample from modified distribution

### Step 4: Implement Detection

No access to ProteinMPNN needed - only need:
- The sequence to test
- Secret key
- γ-generator network

### Step 5: Training Data

Use protein structures from:
- **PDB** (Protein Data Bank)
- **CATH/SCOP** (structural classifications)
- **AlphaFold Database**

Generate training pairs:
- Input: Protein backbone structure
- Output: Sequence (with and without watermark)

## Evaluation Metrics

### 1. **Detectability**
- **True Positive Rate (TPR)**: % of watermarked sequences correctly detected
- **False Positive Rate (FPR)**: % of natural sequences incorrectly flagged
- **Z-score**: Higher = more detectable

### 2. **Protein Quality**
- **perplexity** using ProteinMPNN or ESM
- **ESM embedding similarity**: Cosine similarity between watermarked and original
- **Sequence recovery**: % amino acids matching original
- **Structural metrics** (if folding):
  - RMSD (root mean square deviation)
  - TM-score (template modeling score)

### 3. **Robustness**
- **Mutation attack**: Random amino acid substitutions
- **Insertion/deletion**: Adding/removing amino acids
- **Copy-paste**: Concatenating natural sequences

## Key Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **Small vocabulary (20 AA)** | Lower γ values (e.g., 0.1-0.3) to ensure sufficient green amino acids |
| **Structural constraints** | Use structure-aware semantic loss; train on diverse protein folds |
| **Functional constraints** | Incorporate functional annotations in training; use active site masking |
| **Limited training data** | Pre-train on large protein databases; fine-tune on specific protein families |

## Expected Results

Based on the paper's LLM results, for proteins we expect:

- **Detectability**: 95-100% TPR at 0-1% FPR with appropriate γ and δ
- **Quality preservation**: >0.85 ESM embedding similarity
- **Robustness**: Watermark survives 10-20% random mutations

The key advantage: **token-specific** watermarking adapts to each position's structural/functional constraints, unlike fixed watermarking schemes.

## Practical Applications

1. **Protecting AI-designed proteins**: Identify proteins designed by your model
2. **Tracking protein engineering**: Trace modifications through generations
3. **Preventing misuse**: Detect unauthorized use of your protein design models
4. **Attribution**: Prove ownership of AI-generated protein therapeutics

## Next Steps

1. Implement γ and δ generators (lightweight MLPs)
2. Modify ProteinMPNN to accept watermark parameters
3. Set up training loop with MGDA
4. Evaluate on benchmark protein structures
5. Test robustness against various attacks

---

## References

- **Watermarking Paper**: Huo et al., "Token-Specific Watermarking with Enhanced Detectability and Semantic Coherence for Large Language Models", ICML 2024
- **ProteinMPNN**: Dauparas et al., "Robust deep learning-based protein sequence design using ProteinMPNN", Science 2022
- **ESM**: Lin et al., "Evolutionary-scale prediction of atomic-level protein structure with a language model", Science 2023
