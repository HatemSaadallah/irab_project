# Mathematical Formulation of irab_tashkeel

## 1. Input Encoding

Given input text $t$ of $L$ characters from vocabulary $\mathcal{V}$ ($|\mathcal{V}|=45$):

$$\mathbf{x} = [x_1, x_2, \dots, x_L], \quad x_i \in \mathcal{V}$$

With $W$ whitespace-delimited words, each word $w_j$ spans character indices $[s_j, e_j)$.

## 2. Embedding Layer

$$\mathbf{e}_i = \mathbf{E}_{\text{char}}[x_i] + \mathbf{E}_{\text{pos}}[i], \quad \mathbf{e}_i \in \mathbb{R}^d$$

where $\mathbf{E}_{\text{char}} \in \mathbb{R}^{|\mathcal{V}| \times d}$, $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{L_{\max} \times d}$, and $d$ is the hidden dimension.

$$\mathbf{E}^{(0)} = \text{Dropout}\!\left([\mathbf{e}_1, \dots, \mathbf{e}_L]\right) \in \mathbb{R}^{L \times d}$$

## 3. Transformer Encoder (Pre-Norm)

For layers $\ell = 1, \dots, N$:

$$\mathbf{Z}^{(\ell)} = \mathbf{E}^{(\ell-1)} + \text{MHA}\!\left(\text{LN}\!\left(\mathbf{E}^{(\ell-1)}\right)\right)$$

$$\mathbf{E}^{(\ell)} = \mathbf{Z}^{(\ell)} + \text{FFN}\!\left(\text{LN}\!\left(\mathbf{Z}^{(\ell)}\right)\right)$$

where $\text{LN}$ denotes LayerNorm.

### Multi-Head Attention

$$\text{MHA}(\mathbf{X}) = \text{Concat}(\text{head}_1, \dots, \text{head}_H)\,\mathbf{W}^O$$

$$\text{head}_j = \text{softmax}\!\left(\frac{(\mathbf{X}\mathbf{W}_j^Q)(\mathbf{X}\mathbf{W}_j^K)^\top}{\sqrt{d_k}}\right)\mathbf{X}\mathbf{W}_j^V$$

with $d_k = d / H$, and $\mathbf{W}_j^Q, \mathbf{W}_j^K, \mathbf{W}_j^V \in \mathbb{R}^{d \times d_k}$, $\mathbf{W}^O \in \mathbb{R}^{d \times d}$.

### Feed-Forward Network

$$\text{FFN}(\mathbf{z}) = \text{GELU}(\mathbf{z}\,\mathbf{W}_1 + \mathbf{b}_1)\,\mathbf{W}_2 + \mathbf{b}_2$$

where $\mathbf{W}_1 \in \mathbb{R}^{d \times 4d}$, $\mathbf{W}_2 \in \mathbb{R}^{4d \times d}$.

### Encoder Output

$$\mathbf{H} = \text{LN}\!\left(\mathbf{E}^{(N)}\right) \in \mathbb{R}^{L \times d}$$

## 4. Task Heads

All three heads operate on the shared representation $\mathbf{H}$.

### 4a. Diacritization Head (per-character)

Classifies each character into one of $C_d = 15$ diacritic classes (none, fatḥa, ḍamma, kasra, tanwīn variants, sukūn, shadda combinations):

$$\hat{\mathbf{y}}_i^{\text{diac}} = \mathbf{W}_2^d \; \text{GELU}(\mathbf{W}_1^d \, \mathbf{h}_i + \mathbf{b}_1^d) + \mathbf{b}_2^d$$

$$\hat{\mathbf{y}}_i^{\text{diac}} \in \mathbb{R}^{C_d}, \quad \hat{y}_i = \arg\max_c \; \hat{\mathbf{y}}_{i,c}^{\text{diac}}$$

where $\mathbf{W}_1^d \in \mathbb{R}^{d \times d}$, $\mathbf{W}_2^d \in \mathbb{R}^{d \times C_d}$.

### 4b. I'rāb Head (per-word, with mean pooling)

For each word $w_j$ spanning characters $[s_j, e_j)$, compute a word-level representation by mean-pooling:

$$\bar{\mathbf{h}}_j = \frac{1}{e_j - s_j} \sum_{i=s_j}^{e_j - 1} \mathbf{h}_i$$

Then classify into $C_r = 11$ i'rāb roles (fiil, harf_jarr, N_marfu, N_mansub, ism_majrur, mudaf_ilayh, etc.):

$$\hat{\mathbf{y}}_j^{\text{irab}} = \mathbf{W}_2^r \; \text{GELU}(\mathbf{W}_1^r \, \bar{\mathbf{h}}_j + \mathbf{b}_1^r) + \mathbf{b}_2^r$$

$$\hat{\mathbf{y}}_j^{\text{irab}} \in \mathbb{R}^{C_r}, \quad \hat{r}_j = \arg\max_c \; \hat{\mathbf{y}}_{j,c}^{\text{irab}}$$

where $\mathbf{W}_1^r \in \mathbb{R}^{d \times d/2}$, $\mathbf{W}_2^r \in \mathbb{R}^{d/2 \times C_r}$.

### 4c. Error Detection Head (per-character, BIO)

Classifies each character into $C_e = 7$ BIO error tags (O, B-hamza, I-hamza, B-taa, I-taa, B-case, I-case):

$$\hat{\mathbf{y}}_i^{\text{err}} = \mathbf{W}_2^e \; \text{GELU}(\mathbf{W}_1^e \, \mathbf{h}_i + \mathbf{b}_1^e) + \mathbf{b}_2^e$$

$$\hat{\mathbf{y}}_i^{\text{err}} \in \mathbb{R}^{C_e}, \quad \hat{e}_i = \arg\max_c \; \hat{\mathbf{y}}_{i,c}^{\text{err}}$$

where $\mathbf{W}_1^e \in \mathbb{R}^{d \times d/2}$, $\mathbf{W}_2^e \in \mathbb{R}^{d/2 \times C_e}$.

## 5. Multi-Task Loss

Each training example carries binary masks $m^{\text{diac}}, m^{\text{irab}}, m^{\text{err}} \in \{0, 1\}$ indicating which heads have supervision.

### Per-Head Losses

Cross-entropy with label smoothing $\epsilon$:

$$\mathcal{L}_{\text{diac}} = \frac{1}{|\mathcal{B}_d|} \sum_{n \in \mathcal{B}_d} \frac{1}{L_n} \sum_{i=1}^{L_n} \text{CE}_\epsilon\!\left(\hat{\mathbf{y}}_{n,i}^{\text{diac}},\; y_{n,i}^{\text{diac}}\right)$$

$$\mathcal{L}_{\text{irab}} = \frac{1}{|\mathcal{B}_r|} \sum_{n \in \mathcal{B}_r} \frac{1}{W_n} \sum_{j=1}^{W_n} \text{CE}\!\left(\hat{\mathbf{y}}_{n,j}^{\text{irab}},\; y_{n,j}^{\text{irab}}\right)$$

$$\mathcal{L}_{\text{err}} = \frac{1}{|\mathcal{B}_e|} \sum_{n \in \mathcal{B}_e} \frac{1}{L_n} \sum_{i=1}^{L_n} \text{CE}_\epsilon\!\left(\hat{\mathbf{y}}_{n,i}^{\text{err}},\; y_{n,i}^{\text{err}}\right)$$

where $\mathcal{B}_d = \{n : m_n^{\text{diac}} = 1\}$, $\mathcal{B}_r = \{n : m_n^{\text{irab}} = 1\}$, $\mathcal{B}_e = \{n : m_n^{\text{err}} = 1\}$.

### Label-Smoothed Cross-Entropy

$$\text{CE}_\epsilon(\hat{\mathbf{y}}, y) = -(1-\epsilon)\log\hat{p}_y - \frac{\epsilon}{C}\sum_{c=1}^{C}\log\hat{p}_c, \quad \hat{p}_c = \text{softmax}(\hat{\mathbf{y}})_c$$

### Total Loss

$$\mathcal{L} = \alpha \, \mathcal{L}_{\text{diac}} + \beta \, \mathcal{L}_{\text{irab}} + \gamma \, \mathcal{L}_{\text{err}}$$

Default weights: $\alpha = 1.0$, $\beta = 0.5$, $\gamma = 0.3$.

## 6. Optimization

**Optimizer**: AdamW with weight decay $\lambda$:

$$\theta_{t+1} = \theta_t - \eta_t \left(\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) + \lambda \, \theta_t \right)$$

**Learning rate schedule**: Linear warmup + cosine annealing:

$$\eta_t = \begin{cases} \eta_{\max} \cdot \frac{t}{T_w} & t \leq T_w \\ \eta_{\max} \cdot \frac{1}{2}\left(1 + \cos\!\left(\frac{\pi(t - T_w)}{T - T_w}\right)\right) & t > T_w \end{cases}$$

where $T_w$ is warmup steps and $T$ is total steps.

**Gradient clipping**: $\|\nabla \mathcal{L}\|_2 \leq g_{\max}$ (default $g_{\max} = 1.0$).

## 7. Inference

### 7a. Predictions and Confidence

$$\hat{p}_{i,c}^{\text{diac}} = \text{softmax}(\hat{\mathbf{y}}_i^{\text{diac}})_c, \quad \hat{d}_i = \arg\max_c \; \hat{p}_{i,c}^{\text{diac}}$$

Per-word diacritization confidence (mean over word's characters):

$$\text{conf}_j^{\text{diac}} = \frac{1}{e_j - s_j} \sum_{i=s_j}^{e_j-1} \max_c \; \hat{p}_{i,c}^{\text{diac}}$$

I'rāb confidence:

$$\text{conf}_j^{\text{irab}} = \max_c \; \text{softmax}(\hat{\mathbf{y}}_j^{\text{irab}})_c$$

### 7b. Tier Classification (Rule-Based)

$$\tau = \begin{cases} 3 & \text{if } \exists \; w \in \{\text{الذي, إذا, لو, ...}\} \text{ (relative/conditional)} \\ 2 & \text{if } \exists \; w \in \{\text{كان, إنّ, لم, ...}\} \text{ (kāna/inna/jussive)} \\ 1 & \text{otherwise (simple sentence)} \end{cases}$$

## 8. Evaluation Metrics

### Diacritic Error Rate (DER)

$$\text{DER} = \frac{\sum_{i=1}^{L} \mathbb{1}[\hat{d}_i \neq d_i^*] \cdot \mathbb{1}[d_i^* \neq 0]}{\sum_{i=1}^{L} \mathbb{1}[d_i^* \neq 0]}$$

### Word Error Rate (WER)

$$\text{WER} = \frac{1}{W} \sum_{j=1}^{W} \mathbb{1}\!\left[\exists \, i \in [s_j, e_j) : \hat{d}_i \neq d_i^*\right]$$

### I'rāb Accuracy

$$\text{Acc}_{\text{irab}} = \frac{1}{W} \sum_{j=1}^{W} \mathbb{1}[\hat{r}_j = r_j^*]$$

### Error Span F1

For predicted spans $\hat{S}$ and gold spans $S^*$ (exact match on start, end, type):

$$P = \frac{|\hat{S} \cap S^*|}{|\hat{S}|}, \quad R = \frac{|\hat{S} \cap S^*|}{|S^*|}, \quad F_1 = \frac{2PR}{P + R}$$

## 9. Model Configurations

| Config | $d$ | $H$ | $N$ | $L_{\max}$ | Parameters |
|--------|-----|-----|-----|------------|------------|
| Small  | 256 | 8   | 6   | 256        | ~5M        |
| Medium | 768 | 12  | 12  | 512        | 86.69M     |

Where $d$ = hidden dim, $H$ = attention heads, $N$ = transformer layers, $L_{\max}$ = max sequence length.
