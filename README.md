# LoRA-QLoRA-studies

##  Introduction

This project studies **parameter-efficient fine-tuning** under Colab-limited-scale compute constraints. It includes:

-  review of LoRA and QLoRA paper
-  **Manual LoRA implementation** in PyTorch 
-  **Comparative analysis** with PEFT-based LoRA/QLoRA implementations
-  **Empirical evaluation** of how rank, target modules, and quantization affect:
  - Memory usage
  - Training convergence
  - Validation loss

This is ideal for understanding how parameter-efficient fine-tuning works under practical, resource-limited settings.

---

## LoRA: Low-Rank Adaptation

**Low-Rank Adaptation (LoRA)** fine-tunes large pretrained models by freezing the original weights and training only small low-rank matrices instead of updating all parameters.

### How LoRA Works

For a weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA models the task-specific update as:

$$W = W_0 + \Delta W, \qquad \Delta W = \frac{\alpha}{r}BA$$

Where:

- $W_0$ is **frozen** (not updated)
- $A \in \mathbb{R}^{r \times k}$ – low-rank matrix (trainable)
- $B \in \mathbb{R}^{d \times r}$ – low-rank matrix (trainable)
- $r \ll \min(d,k)$ – the low rank (typically 8–64)
- $\alpha$ – scaling factor (hyperparameter)

![LoRA Layer](./images/lora_layer.png)

---

## Quantization Fundamentals

Quantization represents real-valued tensors using a smaller set of discrete values, reducing memory and compute costs.

### Basic Quantization Process

For a real value $x \in \mathbb{R}$:

$$q = \mathrm{Quantize}(x), \qquad \hat{x} = \mathrm{Dequantize}(q)$$
- $x$ is quantized to $q$ (a discrete value), and $q$ is stored
- when computing, $q$ is dequantized back to $\hat{x}$ (an approximation)

**The Role of the Quantization Scale $s$:**

The scale $s$ is a **normalization factor** that maps continuous floating-point values to a discrete integer space. It represents the **step size** in the quantized representation:

- **Quantization**: Divide the original value by $s$ to get a normalized value, then round to an integer
- **Dequantization**: Multiply the integer back by $s$ to reconstruct an approximation of the original value

**Why a scale is needed:**
- Floating-point values have arbitrary magnitudes (e.g., 0.5, 1.234, 1000.7)
- We need to map them into a bounded integer range (e.g., -128 to 127 for 8-bit, -8 to 7 for 4-bit)
- The scale adapts to the data's magnitude, ensuring all values fit in the target range

**Symmetric quantization** uses a single scale $s$:

$$q = \mathrm{round}\left(\frac{x}{s}\right), \qquad \hat{x} = s \cdot q$$

The scale is typically chosen as:

$$s = \frac{\max(|x|)}{q_{\max}}$$

where $q_{\max}$ is the largest representable integer (e.g., 127 for 8-bit signed, 7 for 4-bit signed).

**Concrete Example: 8-bit Quantization**

Given a list of float values:
$$\mathbf{x} = [0.5, 1.234, -2.1, 3.5, 0.1, -1.8, 2.7, -0.3]$$

For **8-bit signed integers**, the range is $[-128, 127]$ (so $q_{\max} = 127$).

**Step 1:** Find the scale

$$\max(|x|) = \max(0.5, 1.234, 2.1, 3.5, 0.1, 1.8, 2.7, 0.3) = 3.5$$

$$s = \frac{3.5}{127} \approx 0.0276$$

**Step 2:** Quantize each value
$$q_i = \mathrm{round}\left(\frac{x_i}{s}\right)$$

| Original | $x / s$ | Quantized $q$ |
|----------|---------|---------------|
| 0.5      | 18.1    | 18            |
| 1.234    | 44.7    | 45            |
| -2.1     | -76.1   | -76           |
| 3.5      | 126.8   | 127           |
| 0.1      | 3.6     | 4             |
| -1.8     | -65.2   | -65           |
| 2.7      | 97.8    | 98            |
| -0.3     | -10.9   | -11           |

**Step 3:** Dequantize back to float
$$\hat{x}_i = s \cdot q_i$$

| Quantized $q$ | Reconstructed $\hat{x}$ | Error $x - \hat{x}$ |
|---------------|------------------------|---------------------|
| 18            | 0.497                  | 0.003               |
| 45            | 1.242                  | -0.008              |
| -76           | -2.098                 | -0.002              |
| 127           | 3.502                  | -0.002              |
| 4             | 0.110                  | -0.010              |
| -65           | -1.794                 | -0.006              |
| 98            | 2.704                  | -0.004              |
| -11           | -0.304                 | 0.004               |

**Key observations:**
- Quantized values fit in 8-bit integer range (only 1 byte each instead of 4 or 8 bytes for float)
- Reconstruction error is small (< 0.01 per value)
- The scale adapts to the data magnitude (0.0276 in this case)

With value clamping:

$$q = \mathrm{clip}\left(\mathrm{round}\left(\frac{x}{s}\right), q_{\min}, q_{\max}\right)$$

This clips the quantized value within $[q_{\min}, q_{\max}]$ to handle outliers.

**Quantization error**:

$$e = x - \hat{x}$$

In uniform quantizers, error is typically bounded by ~half a step size. Fewer bits = more quantization error, but less memory.
Therefore, when you do quantization, you need to think of performance-memory tradeoff considering how much correctness is required with given memory size.

---

## Why 16-bit Computation Instead of 32-bit?

**FP32** provides high numerical precision but uses significant memory and bandwidth. For large language models, this becomes a major bottleneck.

**Mixed-precision training** stores or computes in lower precision (16-bit formats) for most storage and computation to reduce memory use and move data more efficiently. It keeps the training stable by maintaining a full-precision master copy of gradients while computing forward/backward passes in reduced precision.

---

## Floating-Point Formats

A floating-point number has three components:

- **Sign**: positive or negative
- **Exponent**: the scale/magnitude
- **Fraction (Mantissa)**: precision bits

### FP32 (32-bit float)

```
  +---+----------------+----------------+
  | S |    Exponent    |    Fraction    |
  +---+----------------+----------------+
    1        8               23
```

### FP16 (16-bit float)

```
  +---+-------------------+-------------------+
  | S |     Exponent      |      Fraction     |
  +---+-------------------+-------------------+
    1          5                 10
```

**vs. FP32:**
- ❌ Smaller representable range (fewer exponent bits)
- ❌ Lower precision (fewer fraction bits)
- ✅ Less memory required

### BF16 (Brain Floating Point)

```
  15                    7                      0
  +---+-------------------+--------------------+
  | S |     Exponent      |      Fraction      |
  +---+-------------------+--------------------+
    1          8                  7
```

**Key difference:**
- ✅ **Same exponent size as FP32** → preserves numeric range
- ❌ Fewer fraction bits → lower precision

**When BF16 is useful:**
- Range is critical
- Memory/compute reduction is needed
- Many large model frameworks default to BF16 including Llama-3.2-1B model, the one I fine-tuned.

---

## From Floating-Point to 4-bit Quantization

QLoRA applies quantization aggressively: **pretrained weights are stored in 4-bit**, but are **dequantized to 16-bit for actual computation**.

### QLoRA Workflow

$$\text{Store in 4-bit} \rightarrow \text{Dequantize to 16-bit} \rightarrow \text{Compute in 16-bit}$$

This combines:
- Frozen 4-bit quantized pretrained weights (compact storage)
- Low-rank LoRA adapters (trainable)
- Dramatic memory savings with minimal accuracy loss

---

## Blockwise Quantization

A single global scale can be inefficient—one outlier forces a large scale, wasting precision elsewhere. **Blockwise quantization** divides the tensor into blocks and computes a separate scale per block.

For tensor $X$ split into blocks $X^{(1)}, X^{(2)}, \dots$:

$$q^{(b)} = \mathrm{round}\left(\frac{X^{(b)}}{s_b}\right), \qquad \hat{X}^{(b)} = s_b \cdot q^{(b)}$$

Each block adapts to its own value range, providing better fidelity than global quantization.

---

## How Quantization Scales Are Chosen

The choice of quantization scale $s_b$ is critical for minimizing information loss. The QLoRA paper discusses several approaches:

### 1. Absolute Maximum (AbsMax) Method

The simplest approach uses the absolute maximum value in each block:

$$s_b = \frac{\max(|X^{(b)}|)}{q_{\max}}$$

Where $q_{\max}$ is the maximum representable quantized value (e.g., 7 for 4-bit signed, 15 for 4-bit unsigned).

**Pros:**
- Simple to compute
- Symmetric scaling around zero
- Preserves the full range

**Cons:**
- Sensitive to outliers
- Wastes precision if one extreme value dominates

### 2. Percentile Method

To reduce outlier sensitivity, scales can be computed using percentiles:

$$s_b = \frac{\mathrm{percentile}(|X^{(b)}|, p)}{q_{\max}}$$

Common choices: 99.9th or 99.95th percentile.

**Pros:**
- Robust to outliers
- Better precision utilization for the majority of values

**Cons:**
- Sacrifices precision at the tails
- Slight information loss for extreme values

### 3. Quantile-Based Selection (QLoRA Standard)

QLoRA's approach uses **quantile-aligned scales** combined with the NormalFloat (NF4) codebook:

- For **NF4 codebook**: scales are chosen such that the 16 code points align with quantiles of the standard normal distribution
- The scale $s_b$ is determined from the data distribution to match expected value ranges

This aligns the quantization grid with actual weight distributions rather than arbitrary uniform spacing.

$$s_b = \frac{q_{99.95}}{q_{\text{max}}}$$

Where $q_{99.95}$ is the 99.95th percentile of absolute values in the block.

**Why this works:**
- Transformer weights follow approximately normal distributions
- Allocates more precision to high-probability regions
- Leaves tail regions slightly quantized but acceptable

---

## ⚠️ Why Plain 4-bit Is Hard

4-bit quantization has only $2^4 = 16$ possible codes—an extremely limited alphabet. Uniformly spaced codes lead to large quantization errors, especially problematic for pretrained weights which have very specific statistical properties.

**Key insight**: Pretrained transformer weights are approximately **normally distributed and zero-centered**, so a 4-bit codebook should match this distribution rather than use uniform spacing.

### Pretrained Weights Follow a Gaussian Distribution

One of the most important empirical observations in deep learning is that **pretrained neural network weights naturally follow an approximately normal (bell-curve) distribution**. This is not by accident—it emerges from the initialization schemes (like Xavier/He initialization) and training dynamics.

For pretrained transformer weights:
$$W \sim \mathcal{N}(0, \sigma_{\text{global}}^2)$$

This Gaussian structure is fundamental to understanding why QLoRA works so effectively. The distribution is:
- **Centered at zero** (zero-mean)
- **Exponentially concentrated near the center**
- **Sparse in the tails** (very few extreme values)

This property is widely recognized across deep learning literature, including the QLoRA paper (Dettmers et al., 2023) and foundational work on neural network initialization. When weights follow this normal distribution, quantization schemes that allocate *more precision to high-probability regions* (near zero) and *less precision to rare tail regions* become information-theoretically optimal.

---

## Deep Dive: Quantile Quantization & The Evolution of NF4

To understand why QLoRA's quantization is so unique, we must first understand the general concept of Quantile Quantization, why it is information-theoretically perfect, and why it is normally too expensive to use in practice.

### 1. General Quantile Quantization: What is a Quantile?

In **standard linear quantization** (like INT8), the continuous range of a tensor is divided into completely equal-sized intervals. If your data is uniformly distributed, this is ideal. However, neural network weights are **dense near the center and sparse at the tails**. Equal-sized intervals waste precious bits on empty tail spaces while crowding the highly active center.

**Quantile Quantization** solves this by using the Empirical Cumulative Distribution Function (ECDF) of the data. Instead of keeping the interval widths equal, it forces the amount of data inside each bin to be equal.

**What is a Quantile?**

A quantile is simply the cut-point that divides a sorted distribution into equal probability masses. For a $k$-bit data type (where we have $2^k$ available discrete bins), we must find $2^k - 1$ quantiles such that:

$$P(X \le q_i) = \frac{i}{2^k}$$

**Visualization: Quantile Quantization vs. Uniform Quantization**

Consider weight values drawn from a normal distribution:

```
Uniform Quantization (Equal-Width Bins):
═══════╪═══════╪═══════╪═══════╪═══════╪═══════╪═══════╪═══════╪
Value: -3     -2     -1      0      1      2      3      4
Bin sizes: All equal (~1.0 unit each)
Problem: Most data (near 0) gets crowded; tails waste bins

Quantile Quantization (Equal-Probability Bins):
══╪═════╪═══════════╪═══════════╪═════╪═╪
Value: -3  -2   -1       0       1   2   3
Bin sizes: Narrow near center, wide at tails
Benefit: Each bin represents roughly equal amounts of data
```

This visualization shows that quantile quantization adaptively allocates more precision (narrower bins) where data is dense and less precision where data is sparse.

**The Bottleneck of General Quantile Estimation**

In a standard training pipeline, doing true empirical quantile quantization requires the system to calculate the exact quantiles of live tensors on the fly. This means the hardware must:

1. Copy or stream the entire weight matrix
2. Sort the values or compute a high-resolution histogram to estimate the ECDF
3. Compute the custom bin boundaries for that specific tensor

**Why it is not done in practice**: This process introduces a massive computational bottleneck ($O(N \log N)$ for sorting). Furthermore, fast streaming approximations of quantiles (like T-Digest) introduce approximation errors. In deep learning, compounding rounding errors across dozens of layers can severely degrade a model's performance.

### 2. The Inductive Bias: Why LLM Blocks are Gaussian

QLoRA completely bypasses the expensive runtime sorting of empirical quantile estimation by exploiting a powerful structural property of pre-trained Large Language Models: **weights follow an approximately normal distribution**.

For a pretrained LLM with global weight distribution:
$$W_{\text{global}} \sim \mathcal{N}(0, \sigma_{\text{global}}^2)$$

The key insight is that this Gaussian structure is **inherited locally by small blocks of weights**.

#### The 64-Weight Sub-Sample Behavior

When we implement blockwise quantization with a block size of 64, we are slicing this giant Gaussian ocean into micro-tensors of 64 contiguous weights.

**Statistically**, extracting a block of 64 weights is equivalent to taking a sample of size $n=64$ from a normal population. According to the **Law of Large Numbers and asymptotic normality**, a sample size of 64 is large enough for the sample distribution to preserve the parent population's distribution shape.

Therefore, **every individual block of 64 weights retains a zero-centered normal distribution shape**. The only property that changes from Block $A$ to Block $B$ is the spread—their standard deviation ($\sigma_b$):

$$X^{(b)} \sim \mathcal{N}(0, \sigma_b^2)$$

**Illustration:**

```
Global Weight Distribution (Gaussian):
     ╱╲
    ╱  ╲
   ╱    ╲
  ╱______╲  ← Standard deviation σ_global

Block 1 (64 weights, σ_b1):     Block 2 (64 weights, σ_b2):
    ╱╲                              ╱╲
   ╱  ╲                            ╱  ╲
  ╱____╲  ← σ_b1                  ╱____╲  ← σ_b2
  
Both are normal, just different spreads
```

This elegant property allows QLoRA to avoid sorting entirely—**every block is guaranteed to be roughly normal, just with different variances**.

### 3. Step-by-Step Construction of the NF4 Solution

Because calculating the true statistical variance ($\sigma_b = \sqrt{\frac{1}{N}\sum x_i^2}$) requires squares, reductions, and square roots that slow down GPU kernels, QLoRA replaces it with a hardware-friendly mathematical shortcut.

Here is how the **4-bit NormalFloat (NF4) data type** is mathematically constructed and applied, step by step:

#### Step A: Establishing the $[-1, 1]$ Analytical Sandbox

Instead of changing the 4-bit bin locations for every block's unique $\sigma_b$, we define a single, unchanging **"master grid"** based on a theoretical Standard Normal Distribution $\mathcal{N}(0,1)$. To map this infinite distribution into digital storage, we bound the data type to an arbitrary symmetrical range of $[-1, 1]$. Both our theoretical grid and our live weights must be normalized into this exact sandbox.

#### Step B: Slicing the Theoretical Grid

To build a 4-bit ($2^4 = 16$ bins) quantile data type, we divide the area under the standard normal curve into 16 equal chunks of probability mass. This requires 17 boundary lines. The mathematical center of mass for the $i$-th bin is calculated analytically using the **Gaussian Quantile Function** $Q_X(\cdot)$ (the inverse CDF):

$$\tilde{q}_i = \frac{1}{2} \left[ Q_X\left(\frac{i}{17}\right) + Q_X\left(\frac{i+1}{17}\right) \right]$$

**Visual Example of the 16 Bins:**

```
Standard Normal Distribution N(0,1):

Probability mass divided into 16 equal chunks (6.25% each):

        ╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲
       ╱  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  ╲
      ╱   │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │   ╲
     ╱    └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘    ╲
    ╱────────────────────────────────────────────────────────╲
   ╱                                                            ╲
  ────────────────────────────────────────────────────────────
  -4     -3     -2     -1      0      1      2      3      4
  
Narrower bins near 0     Wider bins at tails
(high density)           (low density)

Each bin represents 6.25% of the probability mass
```

The 16 code point values (quantile bin centers) are placed asymmetrically to match the normal distribution's shape.

#### Step C: The Zero-Symmetry Correction

Because 16 is an even number, a purely symmetric split across a normal distribution places the center line ($0.0$) as a boundary between bins, meaning **no single bin represents exactly 0.0**. To fix this, the authors generate **two separate asymmetrical quantile grids**: one for the negative half $[-1, 0]$ and one for the positive half $[0, 1]$, anchoring an exact 0.0 point in the codebook to perfectly preserve padding and zeroed elements.

```
Without correction (0 as boundary):
Bin 7: [..., -0.003]   |   Bin 8: [+0.001, ...]
                       ↑ Gap at 0.0 (bad for sparse weights)

With correction (0 in a bin):
Bin 7: [..., -0.001]
Bin 8: [-0.0001, +0.0001]  ← Contains exactly 0.0
Bin 9: [+0.001, ...]
                       ✓ Perfect for padding and zeros
```

#### Step D: Pre-Scaling the Codebook (Avoiding Runtime Variance Math)

According to **Extreme Value Theory (EVT)**, the expected absolute maximum value ($c$) of a sample size $n=64$ drawn from a Gaussian distribution is directly proportional to its true standard deviation:

$$\sigma \approx k \cdot c$$

where $k$ is a known constant (approximately 0.4049 for $n=64$).

To **avoid making the GPU compute this scaling constant on live data**, the authors shift the algebraic burden onto the static codebook. They take the 16 raw theoretical bin centers ($\tilde{q}_i$) and divide them by their own maximum absolute value:

$$q_i = \frac{\tilde{q}_i}{\max(|\tilde{q}_0|, \dots, |\tilde{q}_{15}|)}$$

This hardcodes the fixed scaling factors directly into a permanent 16-number lookup table spanning exactly $[-1, 1]$.

**The NF4 Codebook (Pre-computed):**

```
Index | Codebook Value | Meaning
------|---|---
  0   | -1.000         | Extreme negative
  1   | -0.720         | Large negative
  2   | -0.540         | Medium-large negative
  ...                  | ...
  7   | -0.070         | Small negative
  8   | +0.070         | Small positive (near zero)
  9   | +0.150         | Bit-larger positive
  ...                  | ...
  14  | +0.650         | Large positive
  15  | +1.000         | Extreme positive
```

This codebook is **fixed, pre-computed once, and stored as a lookup table**. It's never recalculated during training.

#### Step E: Runtime Absolute Maximum Rescaling

During training, when a block of 64 live weights $X^{(b)}$ needs to be quantized, the GPU executes a single, hardware-accelerated pass to find the absolute maximum value of those 64 numbers. This is our quantization scale, $s_b$:

$$s_b = \max(|X^{(b)}|)$$

The GPU then performs a lightning-fast vector division to push the live weights into the $[-1, 1]$ sandbox:

$$X^{(b)}_{\text{normalized}} = \frac{X^{(b)}}{s_b}$$

**Because both the live weights and the theoretical codebook have been normalized by their respective maximums into the exact same $[-1, 1]$ space, they are perfectly aligned.** The GPU simply matches each normalized weight to the nearest entry in the static NF4 codebook, saving the corresponding 4-bit index.

**Example: Quantizing a Block**

```
Live weights in a block:
X^(b) = [0.051, -0.125, 0.089, 0.003, -0.042, ...]

Step 1: Find max absolute value
s_b = max(|0.051|, |-0.125|, |0.089|, |0.003|, |-0.042|, ...) = 0.125

Step 2: Normalize to [-1, 1]
X^(b)_norm = X^(b) / 0.125
           = [0.408, -1.000, 0.712, 0.024, -0.336, ...]

Step 3: Match each to nearest NF4 codebook entry
0.408 → closest to 0.360 (index 11)
-1.000 → exactly -1.000 (index 0)
0.712 → closest to 0.720 (index 14)
0.024 → closest to 0.070 (index 8)
-0.336 → closest to -0.360 (index 5)
...

Step 4: Store as 4-bit indices (1 nibble per weight)
[11, 0, 14, 8, 5, ...] → Store as 4-bit packed integers
```

Only the **scale $s_b$ (32 bits)** and the **16 indices (4 bits each, = 32 bits for 64 weights)** need to be stored.

---

## NF4: NormalFloat 4-bit

**NF4 (NormalFloat 4-bit)** chooses 16 representable values aligned with a standard normal distribution $\mathcal{N}(0,1)$, rather than evenly spaced integers.

Code points are placed at **quantiles of the normal distribution**, so each bin represents equal probability mass.

### Why NF4 Is Better

- **Many weights cluster near zero** → allocate more precision there
- **Fewer weights in tails** → allocate less precision there
- **Better for transformer weights** than uniform Int4 or FP4

This adaptive quantization is essential for effective 4-bit QLoRA.

---

## Double Quantization

Blockwise quantizers store scales for each block. For block size $B$ with 32-bit scales, the overhead is:

$$\frac{32}{B} \text{ bits/parameter}$$

For $B = 64$: $\frac{32}{64} = 0.5$ bits/parameter.

**Double quantization** quantizes the scales themselves. QLoRA reports average overhead:

$$\frac{8}{64} + \frac{32}{64 \times 256} = 0.127 \text{ bits/parameter}$$

This saves $\sim 0.373$ bits/parameter, compressing not just weights but also reconstruction metadata.

---

## QLoRA in One Sentence

**QLoRA fine-tunes a frozen 4-bit quantized pretrained model by dequantizing weights to 16-bit for computation and training low-rank LoRA adapters on the frozen backbone.**

---

## 📚 References

- **Hu et al.** (2021), *LoRA: Low-Rank Adaptation of Large Language Models*  
  https://arxiv.org/abs/2106.09685

- **Dettmers et al.** (2023), *QLoRA: Efficient Finetuning of Quantized LLMs*  
  https://arxiv.org/abs/2305.14314

- **He et al.** (2015), *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*  
  https://arxiv.org/abs/1502.01852  
  (Foundational work on neural network weight distributions and initialization)

- **LeCun et al.** (1998), *Efficient BackProp*  
  https://yann.lecun.com/expl/papers/solla_92.pdf  
  (Early analysis of weight distributions in neural networks)
