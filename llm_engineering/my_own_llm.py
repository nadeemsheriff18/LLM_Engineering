import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────────
#  SIMPLE TRANSFORMER — built from scratch
#  Input : a string  e.g. "the cat sat"
#  Output: probability distribution over vocab
#          (what token comes next?)
# ─────────────────────────────────────────────

# ── 0. Tiny vocabulary ───────────────────────
# In a real LLM this is 50,000+ tokens (BPE).
# We keep it tiny so you can read every number.

VOCAB = ["<pad>", "the", "cat", "sat", "on", "mat", "a", "dog", "ran", "<unk>"]
VOCAB_SIZE  = len(VOCAB)     # 10
D_MODEL     = 8              # embedding dimension  (GPT-3 uses 12288)
N_HEADS     = 2              # number of attention heads  (GPT-3 uses 96)
D_HEAD      = D_MODEL // N_HEADS   # 4  — dimension per head
SEQ_LEN     = None           # determined at runtime from input


# ════════════════════════════════════════════
#  STEP 1 — TOKENISE
#  Convert raw string → list of integer IDs
# ════════════════════════════════════════════

def tokenise(text: str) -> list[int]:
    """
    Split text on spaces, map each word to its vocab index.
    Unknown words → <unk> index.
    """
    tokens = text.lower().split()
    ids = [VOCAB.index(t) if t in VOCAB else VOCAB.index("<unk>")
           for t in tokens]
    print(f"\n{'='*55}")
    print(f"  INPUT TEXT : '{text}'")
    print(f"  TOKENS     : {tokens}")
    print(f"  TOKEN IDs  : {ids}")
    return ids


# ════════════════════════════════════════════
#  STEP 2 — EMBEDDING
#  Token IDs → dense vectors of shape [D_MODEL]
#  In a real model these are LEARNED weights.
#  We use fixed random weights seeded for
#  reproducibility.
# ════════════════════════════════════════════

np.random.seed(42)
EMBEDDING_MATRIX = np.random.randn(VOCAB_SIZE, D_MODEL) * 0.1
# Shape: [10 vocab × 8 dims]
# Each row is the embedding vector for one token.

def embed(token_ids: list[int]) -> np.ndarray:
    """
    Look up each token ID in the embedding matrix.
    Returns shape: [seq_len, d_model]
    """
    embeddings = EMBEDDING_MATRIX[token_ids]      # fancy indexing
    print(f"\n{'─'*55}")
    print(f"  STEP 2 — INPUT EMBEDDINGS")
    print(f"  Shape: {embeddings.shape}  (seq_len × d_model)")
    print(f"  Each row = one token's vector:\n")
    for i, tid in enumerate(token_ids):
        print(f"    '{VOCAB[tid]}' → {embeddings[i].round(3)}")
    return embeddings


# ════════════════════════════════════════════
#  STEP 3 — POSITIONAL ENCODING
#  Inject word ORDER into embeddings using
#  sine/cosine waves (original paper approach).
#
#  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
#  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
#
#  Why this works: different frequencies encode
#  different position ranges — like clock hands
#  at different speeds. The model can learn to
#  read position from these patterns.
# ════════════════════════════════════════════

def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Returns shape: [seq_len, d_model]
    """
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            angle = pos / (10000 ** (i / d_model))
            PE[pos, i]   = np.sin(angle)   # even dimensions
            if i + 1 < d_model:
                PE[pos, i+1] = np.cos(angle)  # odd dimensions
    return PE

def add_positional_encoding(embeddings: np.ndarray) -> np.ndarray:
    seq_len, d_model = embeddings.shape
    PE = positional_encoding(seq_len, d_model)
    result = embeddings + PE
    print(f"\n{'─'*55}")
    print(f"  STEP 3 — POSITIONAL ENCODING ADDED")
    print(f"  Sine/cosine waves added to each embedding.")
    print(f"  Now 'the' at position 0 ≠ 'the' at position 2.")
    print(f"  Positional encodings:\n")
    for pos in range(seq_len):
        print(f"    pos {pos}: {PE[pos].round(3)}")
    return result


# ════════════════════════════════════════════
#  STEP 4 — MULTI-HEAD SELF-ATTENTION
#
#  The heart of the Transformer.
#
#  For each head h:
#    Q_h = X · W_Q_h      [seq, d_head]
#    K_h = X · W_K_h      [seq, d_head]
#    V_h = X · W_V_h      [seq, d_head]
#
#    scores_h = Q_h · K_hᵀ / sqrt(d_head)   [seq, seq]
#    weights_h = softmax(scores_h)            [seq, seq]
#    head_h    = weights_h · V_h             [seq, d_head]
#
#  Concat all heads → [seq, d_model]
#  Project with W_O → [seq, d_model]
#
#  CAUSAL MASK: for decoder (GPT-style),
#  future tokens are masked to -inf so their
#  attention weight becomes 0 after softmax.
# ════════════════════════════════════════════

# Learned projection matrices — one set per head
# In a real model these are learned via backprop.
np.random.seed(7)
W_Q = [np.random.randn(D_MODEL, D_HEAD) * 0.1 for _ in range(N_HEADS)]
W_K = [np.random.randn(D_MODEL, D_HEAD) * 0.1 for _ in range(N_HEADS)]
W_V = [np.random.randn(D_MODEL, D_HEAD) * 0.1 for _ in range(N_HEADS)]
W_O = np.random.randn(D_MODEL, D_MODEL) * 0.1  # output projection

def softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax with numerical stability."""
    x = x - x.max(axis=-1, keepdims=True)   # subtract max for stability
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def causal_mask(seq_len: int) -> np.ndarray:
    """
    Lower-triangular mask of 0s and -inf.
    Token at position i cannot attend to position j > i.

    Example for seq_len=3:
    [[  0, -inf, -inf],
     [  0,   0,  -inf],
     [  0,   0,    0 ]]
    """
    mask = np.zeros((seq_len, seq_len))
    mask[np.triu_indices(seq_len, k=1)] = -np.inf
    return mask

def multi_head_attention(X: np.ndarray, masked: bool = True) -> np.ndarray:
    """
    X shape: [seq_len, d_model]
    Returns: [seq_len, d_model]
    """
    seq_len = X.shape[0]
    mask = causal_mask(seq_len) if masked else np.zeros((seq_len, seq_len))

    print(f"\n{'─'*55}")
    print(f"  STEP 4 — MULTI-HEAD SELF-ATTENTION ({N_HEADS} heads)")
    print(f"  d_head = {D_HEAD}  (d_model {D_MODEL} / n_heads {N_HEADS})")

    head_outputs = []

    for h in range(N_HEADS):
        # ── Project input into Q, K, V ──────────
        Q = X @ W_Q[h]   # [seq, d_head]
        K = X @ W_K[h]   # [seq, d_head]
        V = X @ W_V[h]   # [seq, d_head]

        # ── Attention scores ─────────────────────
        # Dot product: how much does each Q match each K?
        scores = Q @ K.T / np.sqrt(D_HEAD)   # [seq, seq]

        # ── Apply causal mask ────────────────────
        scores = scores + mask   # -inf positions → 0 after softmax

        # ── Softmax → attention weights ──────────
        weights = softmax(scores)   # [seq, seq]  each row sums to 1

        # ── Weighted sum of Values ───────────────
        head_out = weights @ V   # [seq, d_head]
        head_outputs.append(head_out)

        print(f"\n  ┌─ Head {h+1} ─────────────────────────────┐")
        print(f"  │ Q shape: {Q.shape}  K shape: {K.shape}  V shape: {V.shape}")
        print(f"  │ Raw attention scores (before mask):")
        print(f"  │   {scores.round(3)}")
        print(f"  │ Attention WEIGHTS (after softmax + mask):")
        for i, row in enumerate(weights):
            token_weights = "  ".join([f"{w:.2f}" for w in row])
            print(f"  │   '{VOCAB[0]}' pos{i}: [{token_weights}]")
        print(f"  └────────────────────────────────────────┘")

    # ── Concatenate all heads ────────────────────
    # [seq, d_head] × N_HEADS → [seq, d_model]
    concat = np.concatenate(head_outputs, axis=-1)

    # ── Output projection ────────────────────────
    # Mix information across heads
    output = concat @ W_O   # [seq, d_model]

    print(f"\n  Concatenated heads shape: {concat.shape}")
    print(f"  After output projection  : {output.shape}")
    return output


# ════════════════════════════════════════════
#  STEP 5 — ADD & NORM
#  Residual connection + Layer Normalisation
#
#  output = LayerNorm(X + attention_output)
#
#  WHY RESIDUAL: gives gradients a direct path
#  backwards — prevents vanishing over 96 layers.
#
#  WHY LAYERNORM: normalises each token's vector
#  to mean=0, std=1 (then scales with learned
#  gamma/beta). Keeps activations stable.
# ════════════════════════════════════════════

def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std  = x.std(axis=-1,  keepdims=True)
    return (x - mean) / (std + eps)

def add_and_norm(x: np.ndarray, sublayer_output: np.ndarray) -> np.ndarray:
    return layer_norm(x + sublayer_output)


# ════════════════════════════════════════════
#  STEP 6 — FEED-FORWARD NETWORK (FFN)
#  Applied independently to each token.
#
#  FFN(x) = ReLU(x · W1 + b1) · W2 + b2
#
#  Expands d_model → 4×d_model → d_model
#  (8 → 32 → 8 in our tiny model)
#  (768 → 3072 → 768 in GPT-2)
#
#  This is where factual knowledge is believed
#  to be stored — a key-value memory.
# ════════════════════════════════════════════

D_FF = D_MODEL * 4   # 32 in our model, 3072 in GPT-2

np.random.seed(13)
W1 = np.random.randn(D_MODEL, D_FF)  * 0.1
b1 = np.zeros(D_FF)
W2 = np.random.randn(D_FF,   D_MODEL) * 0.1
b2 = np.zeros(D_MODEL)

def feed_forward(x: np.ndarray) -> np.ndarray:
    """
    x shape: [seq_len, d_model]
    Returns: [seq_len, d_model]
    """
    hidden = np.maximum(0, x @ W1 + b1)   # ReLU activation
    output = hidden @ W2 + b2
    print(f"\n{'─'*55}")
    print(f"  STEP 6 — FEED-FORWARD NETWORK")
    print(f"  {D_MODEL} → {D_FF} → {D_MODEL}  (expand × 4 then compress)")
    print(f"  ReLU zeros out negative activations.")
    print(f"  Output shape: {output.shape}")
    return output


# ════════════════════════════════════════════
#  STEP 7 — LANGUAGE MODEL HEAD
#  Project final hidden state → vocab logits
#  then softmax → probability over all tokens.
#
#  logits = X · W_unembed    [seq, vocab_size]
#  probs  = softmax(logits)
#
#  We take the LAST token's row — that's the
#  prediction for "what comes next".
# ════════════════════════════════════════════

np.random.seed(99)
W_unembed = np.random.randn(D_MODEL, VOCAB_SIZE) * 0.1

def lm_head(x: np.ndarray) -> np.ndarray:
    """
    x shape: [seq_len, d_model]
    Returns: [vocab_size] — probability distribution for next token
    """
    logits = x @ W_unembed        # [seq, vocab_size]
    last_token_logits = logits[-1] # take the LAST position only
    probs = softmax(last_token_logits)

    print(f"\n{'─'*55}")
    print(f"  STEP 7 — LANGUAGE MODEL HEAD")
    print(f"  Projecting last token's vector → vocab logits")
    print(f"  Logits shape: {logits.shape}")
    print(f"\n  NEXT TOKEN PROBABILITY DISTRIBUTION:")
    print(f"  {'Token':<12} {'Probability':>12}  {'Bar'}")
    print(f"  {'─'*45}")

    sorted_idx = np.argsort(probs)[::-1]
    for idx in sorted_idx:
        bar = "█" * int(probs[idx] * 80)
        print(f"  {VOCAB[idx]:<12} {probs[idx]:>11.4f}  {bar}")

    predicted_id = np.argmax(probs)
    print(f"\n  → PREDICTED NEXT TOKEN: '{VOCAB[predicted_id]}'  "
          f"(prob: {probs[predicted_id]:.4f})")
    return probs


# ════════════════════════════════════════════
#  FULL FORWARD PASS
#  Chains all steps together.
#  This is one Transformer LAYER (N=1 for clarity)
#  Real models stack this 12–96 times.
# ════════════════════════════════════════════

def forward(text: str) -> np.ndarray:
    print(f"\n{'═'*55}")
    print(f"  TRANSFORMER FORWARD PASS")
    print(f"  d_model={D_MODEL}  n_heads={N_HEADS}  "
          f"d_head={D_HEAD}  vocab={VOCAB_SIZE}")
    print(f"{'═'*55}")

    # 1. Tokenise
    token_ids = tokenise(text)

    # 2. Embed
    X = embed(token_ids)                    # [seq, d_model]

    # 3. Add positional encoding
    X = add_positional_encoding(X)          # [seq, d_model]

    # 4. Multi-head self-attention
    attn_out = multi_head_attention(X, masked=True)

    # 5. Add & Norm (residual around attention)
    X = add_and_norm(X, attn_out)           # [seq, d_model]
    print(f"\n{'─'*55}")
    print(f"  STEP 5 — ADD & NORM (after attention)")
    print(f"  Residual: X = LayerNorm(X + attn_output)")
    print(f"  Shape: {X.shape}  — activations stabilised")

    # 6. Feed-forward network
    ff_out = feed_forward(X)                # [seq, d_model]

    # 7. Add & Norm (residual around FFN)
    X = add_and_norm(X, ff_out)             # [seq, d_model]
    print(f"\n  ADD & NORM (after FFN): shape {X.shape}")

    # 8. LM head → next token probabilities
    print(f"\n{'─'*55}")
    probs = lm_head(X)

    print(f"\n{'═'*55}")
    print(f"  MATRIX SHAPES SUMMARY")
    print(f"{'═'*55}")
    seq = len(token_ids)
    print(f"  Input tokens         : [{seq}]")
    print(f"  After embedding      : [{seq} × {D_MODEL}]")
    print(f"  After pos encoding   : [{seq} × {D_MODEL}]")
    print(f"  Q, K, V per head     : [{seq} × {D_HEAD}]")
    print(f"  Attention weights    : [{seq} × {seq}]  per head")
    print(f"  After attention      : [{seq} × {D_MODEL}]")
    print(f"  After FFN            : [{seq} × {D_MODEL}]")
    print(f"  Final logits         : [{seq} × {VOCAB_SIZE}]")
    print(f"  Next token probs     : [{VOCAB_SIZE}]")
    print(f"{'═'*55}\n")

    return probs


# ════════════════════════════════════════════
#  RUN IT
# ════════════════════════════════════════════

if __name__ == "__main__":
    probs = forward("the cat sat")

    # You can try other inputs too:
    # forward("a dog ran")
    # forward("the mat")