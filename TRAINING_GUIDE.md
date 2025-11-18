# Training Guide: Fidelity-Based Watermark Generators

## Overview

We now have **two training approaches** that use ProteinMPNN fidelity scores as the quality metric (semantic loss), as suggested by your professor.

## Training Scripts

### 1. `train_with_fidelity_reinforce.py` (Recommended)

**Method**: REINFORCE policy gradient algorithm

**How it works**:
1. Generate watermarked sequences using current γ/δ generators
2. Compute fidelity scores from ProteinMPNN (how well sequence fits structure)
3. Compute detectability scores (z-scores)
4. Combined reward: `R = α_fidelity × fidelity + α_detect × z_score`
5. Update generators using policy gradients to maximize reward

**Pros**:
- Theoretically principled for non-differentiable sampling
- Directly optimizes the actual objective (fidelity + detectability)
- Aligns with ICML paper's semantic loss approach

**Cons**:
- Slower (generates sequences each iteration)
- Higher variance (mitigated with baseline)

**To run**:
```bash
python train_with_fidelity_reinforce.py
```

**Expected output**:
```
Training...
Epoch    Reward       Fidelity     Z-score    Loss         Delta      Gamma
--------------------------------------------------------------------------------
1        -1.2345      -2.3456      0.50       0.1234       1.5000     0.5000
5        0.1234       -1.5000      1.20       0.0567       2.3000     0.5200
...
50       2.5000       -0.8000      3.40       0.0123       3.2000     0.4800
```

### 2. `train_with_fidelity.py` (Alternative)

**Method**: Hybrid approach with partial fidelity integration

**How it works**:
1. Uses differentiable surrogate losses for detection (as before)
2. Generates sequences with Gumbel-Softmax (partially differentiable)
3. Computes ProteinMPNN fidelity scores
4. Combines both objectives

**Pros**:
- Faster training
- Lower variance gradients

**Cons**:
- Fidelity gradients don't fully flow back to generators
- Less aligned with true objective

**To run**:
```bash
python train_with_fidelity.py
```

## What is Fidelity Score?

**Fidelity** = Average log probability that ProteinMPNN assigns to a sequence given the structure

- Higher fidelity = sequence fits structure better = higher quality
- ProteinMPNN was trained to predict native sequences from structures
- If a watermarked sequence has low fidelity, it likely has poor structural/functional properties

**Example**:
```
Structure: 5L33 (a real protein structure)
Sequence A: MKTAYIAKQRQ... (native-like)     → Fidelity: -1.2
Sequence B: WWWWWWWWWWW... (all tryptophan)  → Fidelity: -8.5
```

## How This Differs From Previous Training

### Old Approach (train_watermark_generators_simplified.py)
```
Quality Loss = penalty if δ > 5.0 + penalty if γ outside [0.35, 0.65]
```
- Arbitrary constraints
- Doesn't actually measure protein quality
- Just prevents extreme values

### New Approach (fidelity-based)
```
Quality = ProteinMPNN fidelity score (learned from real proteins)
```
- ProteinMPNN judges quality based on what it learned from nature
- Directly measures if sequence is realistic
- This is the "semantic loss" from ICML paper

## Expected Results

After training with fidelity-based approach, you should see:

1. **Higher fidelity scores**: Watermarked sequences should have fidelity closer to natural sequences
2. **Maintained detectability**: Z-scores should still be 2-4 (detectable at 1% FPR)
3. **Better balance**: Generators learn to create strong watermarks without destroying protein quality

**Target metrics**:
- Fidelity: > -2.0 (closer to 0 is better)
- Z-score: 2.5-4.0 (for 95%+ detection)
- Detection @ 1% FPR: ≥ 80%

## Testing After Training

After training completes, evaluate the trained generators:

```bash
# Modify evaluate_trained_generators.py to load the new checkpoint
python evaluate_trained_generators.py
```

Or create a new evaluation script that also reports fidelity:

```bash
# TODO: Create evaluate_fidelity.py that shows:
# - Detection rate
# - Average fidelity of watermarked sequences
# - Average fidelity of baseline sequences
# - Fidelity gap (should be small)
```

## Troubleshooting

### Issue: "CUDA out of memory"
- Reduce `num_samples` in train_step (default: 5)
- Use CPU: The scripts auto-detect device

### Issue: "Fidelity scores are very negative"
- This is normal initially (random generators produce bad sequences)
- Should improve during training
- Native sequences typically have fidelity around -1.0 to -2.0

### Issue: "Z-scores dropping during training"
- Fidelity weight too high - reduce `alpha_fidelity`
- Try: `alpha_fidelity=0.5, alpha_detect=1.0`

### Issue: "Training is very slow"
- REINFORCE generates sequences each iteration (expensive)
- Reduce `num_samples` from 5 to 3
- Reduce `num_epochs` from 50 to 30
- Use GPU if available

## Next Steps

1. **Run training**: `python train_with_fidelity_reinforce.py`
2. **Evaluate**: Create evaluation script with fidelity metrics
3. **Report to professor**: Show fidelity scores along with detection rates
4. **Iterate**: Adjust alpha weights if needed

## Questions to Ask Professor

1. What fidelity score is acceptable? (native sequences: ~-1.5)
2. How to balance fidelity vs detectability weights?
3. Should we test on multiple structures (not just 5L33)?
4. Do you want perplexity instead of fidelity?

---

**Note**: The REINFORCE approach is recommended because it properly handles the non-differentiable sequence generation and directly optimizes what we care about (fidelity + detectability).
