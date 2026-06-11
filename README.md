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

**Quantization error**:

$$e = x - \hat{x}$$

In uniform quantizers, error is typically bounded by ~half a step size. Fewer bits = more quantization error, but less memory.
Therefore, when you do quantization, you need to think of performance-memory tradeoff considering how much correctness is required with given memory size.

---

## Why 16-bit Computation Instead of 32-bit?

**FP32** provides high numerical precision but uses significant memory and bandwidth. For large language models, this becomes a major bottleneck.

**Mixed-precision training** stores or computes in lower precision (16-bit formats) for most storage and computation to reduce memory use and move data more efficiently. It keeps the training stable by carefully managing numerical precision where it matters most.

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

For tensor $X$ split into blocks $X^{(1)}, X^{(2)}, \dots$ (typically 64 weights each):

$$q^{(b)} = \mathrm{round}\left(\frac{X^{(b)}}{s_b}\right), \qquad \hat{X}^{(b)} = s_b \cdot q^{(b)}$$

Each block adapts to its own value range, providing better fidelity than global quantization. By using small block sizes (e.g., 64 weights), we avoid the impact of outliers that would otherwise dominate a global scale calculation.

---

## ⚠️ Why Plain 4-bit Is Hard

4-bit quantization has only $2^4 = 16$ possible codes—an extremely limited alphabet. Uniformly spaced codes lead to large quantization errors, especially problematic for pretrained weights.

**Key insight**: Pretrained transformer weights are approximately **normally distributed and zero-centered**, so a 4-bit codebook should match this distribution rather than use uniform spacing.

---

## Quantile Quantization for NF4

### Information-Theoretic Optimality: Why Quantiles?

**Quantile-based quantization** is information-theoretically optimal for a given data distribution. The key principle is:

> **Assign an equal number of values from the input tensor to each quantization bin.**

This is the optimal data type that minimizes quantization error for a given number of bits. Instead of spacing code points uniformly (which wastes precision), quantile-based quantization places code points where they matter most in the actual data distribution.

For example, if you have a normal distribution with most values near zero:
- Uniform spacing wastes bits on rare tail values
- Quantile spacing allocates more codes near zero (where data concentrates) and fewer in the tails

### The Challenge: Computing Quantiles

In theory, the optimal quantile-based scale could be computed by:

1. **Estimating the Empirical Cumulative Distribution Function (ECDF)**: Sort all values in a block and find values at specific percentiles
2. **Standardization by standard deviation**: Scale all weights to a standard distribution so one quantile set works for all blocks

However, these are **computationally expensive**:
- Sorting or scanning for percentiles adds overhead during inference
- Computing standard deviation for each block requires extra computation

### QLoRA's Solution: One Fixed Quantile Set for Normal Distributions

QLoRA's key insight is to use **a single fixed set of quantiles** that work for all blocks, exploiting the structure of neural network weights.

**Observation**: Pretrained neural network weights (especially in transformers) are approximately **zero-centered normal distributions**, $X \sim \mathcal{N}(0, \sigma)$, where $\sigma$ varies per block. By dividing each block's weights by its standard deviation $\sigma$, we standardize all blocks to $\mathcal{N}(0, 1)$.

Therefore, we can:
1. Compute quantiles **once** for the standard normal distribution $\mathcal{N}(0, 1)$
2. Apply these same quantiles to **all blocks** (after normalization)
3. Store only the block's standard deviation $\sigma$ (not full block-specific quantile tables)

This avoids expensive per-block quantile calculations while achieving information-theoretic optimality.

### QLoRA's Formal Approach

QLoRA transforms weights into a **standardized range $[-1, 1]$** and computes quantiles accordingly. Here's the step-by-step procedure:

#### Step 1: Estimate Quantiles of the Standard Normal Distribution

For $k$-bit quantization, compute $2^k$ quantile values from $\mathcal{N}(0, 1)$ to obtain a $k$-bit quantile quantization data type for normal distributions:

$$q_i = \frac{1}{2}\left( Q_X\left(\frac{i}{2^k + 1}\right) + Q_X\left(\frac{i+1}{2^k + 1}\right) \right)$$

where $Q_X(\cdot)$ is the quantile function (inverse CDF) of the standard normal distribution $\mathcal{N}(0, 1)$.

For **4-bit quantization** ($k=4$, so $2^k = 16$ code points), the quantiles are:

$$q_i = \frac{1}{2}\left( Q_X\left(\frac{i}{17}\right) + Q_X\left(\frac{i+1}{17}\right) \right) \quad \text{for } i = 0, 1, \ldots, 15$$

This gives 16 code points placed at **equal-probability intervals** of the standard normal:

$$\{-1.80, -1.47, -1.23, -1.04, -0.83, -0.68, -0.55, -0.42, 0.42, 0.55, 0.68, 0.83, 1.04, 1.23, 1.47, 1.80\}$$

(These values are symmetric and more densely packed near zero, following the normal distribution.)

#### Step 2: Normalize Quantiles to the Data Range $[-1, 1]$

Normalize these quantiles so they fit exactly in the range $[-1, 1]$:

$$\tilde{q}_i = \frac{q_i}{\max(|q|)}$$

This ensures the data type and input weights can be mapped to the same range.

#### Step 3: Normalize Input Weights by Absolute Maximum

For each weight block $X^{(b)}$, normalize to the range $[-1, 1]$ using **absolute maximum scaling**:

$$\tilde{X}^{(b)} = \frac{X^{(b)}}{\max(|X^{(b)}|)}$$

This rescaling ensures weights fit in the normalized range $[-1, 1]$ to match the quantile data type.

#### Step 4: Match Weight Standard Deviations to Quantile Data Type

The above process is equivalent to rescaling the weight tensor's standard deviation to match the data type's standard deviation. More formally:

> **For zero-mean normal distributions with arbitrary standard deviations $\sigma$, step 3 is equivalent to dividing by $\sigma$ to obtain a standardized distribution**, then applying the fixed quantiles.

The absolute maximum rescaling achieves similar standardization without explicitly computing $\sigma$ for every block.

#### Step 5: Quantize Using Nearest Quantile

For each normalized weight $\tilde{x} \in [-1, 1]$, find the nearest quantile and store its code index (0-15):

$$c = \arg\min_i |\tilde{x} - \tilde{q}_i|$$

Store only the 4-bit code $c$.

#### Step 6: Dequantization (Forward/Backward Pass)

During computation, dequantize by:

1. Retrieve the 4-bit code $c$
2. Look up the normalized quantile $\tilde{q}_c$ from the fixed table
3. Rescale back to the block's magnitude: $\hat{x} = \max(|X^{(b)}|) \cdot \tilde{q}_c$

### Why This Works: Information-Theoretic Optimality

1. **Fixed quantile set**: The 16 quantiles are precomputed once for all blocks, avoiding per-block overhead
2. **Matches weight distribution**: Since weights are approximately normal, quantiles of $\mathcal{N}(0, 1)$ align with the actual weight distribution
3. **Allocates precision optimally**: Codes are denser near zero (where weights cluster) and sparser in the tails (where weights are rare)
4. **Efficient normalization**: Using absolute maximum is faster than computing standard deviation on every block

### Handling Zero Exactly

A challenge with symmetric $k$-bit quantization is the lack of an exact representation of zero, which is important for padding and other zero-valued elements. QLoRA addresses this by including zero as one of the $2^k$ code points, ensuring zero values are represented exactly.

---

## NF4: NormalFloat 4-bit

**NF4 (NormalFloat 4-bit)** is the specific 4-bit quantization data type used in QLoRA. It places 16 code points at quantiles of the standard normal distribution $\mathcal{N}(0,1)$, rather than evenly spaced integers.

### Why NF4 Is Better

- **Many weights cluster near zero** → allocate more codes there
- **Fewer weights in tails** → allocate fewer codes there
- **Information-theoretically optimal** for normal distributions
- **Better for transformer weights** than uniform Int4 or FP4

The NF4 data type is specifically designed for pretrained neural network weights and is essential for effective 4-bit QLoRA.

---

## Double Quantization

Blockwise quantizers store the absolute maximum value (scale) for each block. For block size $B$ with 32-bit scales, the overhead is:

$$\frac{32}{B} \text{ bits/parameter}$$

For $B = 64$: $\frac{32}{64} = 0.5$ bits/parameter.

**Double quantization** quantizes the scales themselves. Instead of storing full 32-bit scales, QLoRA quantizes the scales to 8-bit using a second level of quantization. QLoRA reports average overhead:

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
