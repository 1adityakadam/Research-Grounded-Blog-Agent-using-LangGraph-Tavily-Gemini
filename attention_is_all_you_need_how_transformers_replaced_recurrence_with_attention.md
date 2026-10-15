# Attention Is All You Need: How Transformers Replaced Recurrence with Attention

## Start with the problem: sequence modeling without recurrence

A sequence is more than a collection of independent tokens: meaning often depends on relationships between tokens far apart. In a sentence, a pronoun may refer to a noun several words earlier, or a word’s interpretation may depend on a phrase near the end.

Recurrent neural networks process a sequence one step at a time. At each position, the model reads the current token and updates a hidden state that carries information forward. This creates two limitations: computation is sequential, preventing efficient parallel processing during training, and information from distant positions must pass through many recurrent updates.

Convolutions can expose a token to nearby context in parallel, but modeling relationships across an entire sequence requires multiple layers, wider kernels, or additional mechanisms. The Transformer asks a more direct question: instead of passing information along a chain, can each token learn which other tokens matter and interact with them directly?

Attention provides that mechanism. For every token, the model computes weighted connections to other tokens, allowing relevant context to be gathered in a small number of operations. Because these interactions can be calculated for all positions at once, the architecture is highly parallelizable during training.

The original Transformer organizes this idea as an encoder-decoder stack. Encoder layers combine self-attention with position-wise feed-forward networks. Decoder layers use masked self-attention, cross-attention to the encoder’s representations, and another feed-forward network. Positional information supplies sequence order, while attention supplies content-dependent communication.

## Build the self-attention mechanism step by step

Self-attention lets every token build a representation that incorporates information from other tokens in the same sequence. The mechanism does this with three learned projections: **queries**, **keys**, and **values**.

Given an input matrix \(X\), where each row represents one token embedding, the model computes:

\[
Q = XW_Q,\qquad K = XW_K,\qquad V = XW_V
\]

A **query** describes what information a token is looking for. A **key** describes what information each token offers for matching. A **value** contains the representation that will actually be passed along if a token is considered relevant.

### Scaled dot-product attention

The core operation is:

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\mathsf{T}}{\sqrt{d_k}}\right)V
\]

Each query is compared with every key, the scores are scaled, softmax converts them into weights, and those weights produce a weighted sum of the values.

![Diagram showing queries compared with keys, scaled scores passed through softmax, and weighted values combined into context vectors](images/scaled_dot_product_attention_flow.png)
*Scaled dot-product attention turns query-key similarities into weighted mixtures of values.*

For a token \(i\), the process is:

1. Compare its query \(q_i\) with every key \(k_j\).
2. Divide the scores by \(\sqrt{d_k}\).
3. Apply softmax to obtain attention weights \(\alpha_{ij}\).
4. Compute a weighted sum of all values:

\[
z_i = \sum_j \alpha_{ij}v_j
\]

Scaling matters because dot products tend to grow as vector dimensions grow. Without scaling, softmax can concentrate almost all probability on one token, producing tiny gradients for other positions.

Consider the sentence **“The animal didn’t cross the street because it was tired.”** When processing “it,” self-attention can compare its query with keys for every word. The key for “animal” may receive a stronger match than the key for “street,” so the output for “it” becomes mostly informed by the value for “animal” while still incorporating other context.

A minimal implementation is:

```python
def attention(Q, K, V):
    scores = Q @ K.T
    scores /= (K.shape[-1] ** 0.5)
    weights = softmax(scores, axis=-1)
    return weights @ V
```

In self-attention, \(Q\), \(K\), and \(V\) all come from the same sequence. Multi-head attention repeats this process with separate projections, while cross-attention obtains queries from one sequence and keys and values from another.

## Multi-Head Attention: Multiple Relationship Detectors

A single attention operation must compress many kinds of relationships into one set of attention scores. Multi-head attention gives the model several smaller attention mechanisms, or **heads**, that examine the same sequence from different learned perspectives.

For an input matrix \(X \in \mathbb{R}^{n \times d_{\text{model}}}\), each head independently projects \(X\) into queries, keys, and values:

\[
Q_i = XW_i^Q,\qquad K_i = XW_i^K,\qquad V_i = XW_i^V
\]

Each head performs scaled dot-product attention. One head might focus on local syntax, another on pronoun resolution, and another on long-range semantic dependencies. These behaviors are learned rather than manually assigned.

The outputs are concatenated and mixed by a final projection:

\[
\text{MultiHead}(X)
=
\text{Concat}(\text{head}_1,\ldots,\text{head}_h)W^O
\]

With \(h\) heads and \(d_k=d_v=d_{\text{model}}/h\), concatenation restores an \(n \times d_{\text{model}}\) representation.

## Restore token order with positional information

Self-attention compares every token with every other token, but the operation itself does not know where tokens appeared. Without a position signal, self-attention is permutation-equivariant: it preserves relationships among items but cannot distinguish different orderings of the same token representations.

The original Transformer combines each token embedding with a positional encoding:

\[
x_p = e_p + s_p
\]

The original architecture used fixed sinusoidal signals:

\[
s_{p,2i}=\sin\left(p/10000^{2i/d}\right),\qquad
s_{p,2i+1}=\cos\left(p/10000^{2i/d}\right)
\]

Different dimensions oscillate at different rates, giving each position a distinctive pattern. Learned positional embeddings instead use trainable vectors for positions. Fixed signals provide a structured representation that can be generated for longer sequences, while learned embeddings offer flexibility within a defined range.

## Trace the encoder-decoder Transformer architecture

The original Transformer is an encoder-decoder system designed for sequence-to-sequence tasks such as translation. The encoder reads the entire source sentence and builds contextual representations. The decoder generates the target sentence one token at a time using previously generated tokens and the encoder’s output.

![Encoder-decoder Transformer diagram showing masked self-attention, cross-attention, feed-forward layers, residual connections, and positional information](images/transformer_encoder_decoder_architecture.png)
*The original Transformer pairs an encoder stack with a decoder stack connected by cross-attention.*

An **encoder layer** contains self-attention followed by a position-wise feed-forward network. Residual connections and normalization surround these sublayers. The encoder stacks several such layers, allowing representations to become progressively more contextual.

A **decoder layer** has masked self-attention, encoder-decoder **cross-attention**, and a position-wise feed-forward network. In cross-attention, the decoder supplies queries while the encoder supplies keys and values:

\[
Q=DW_Q,\qquad K=EW_K,\qquad V=EW_V
\]

Thus, each target position asks which source positions are relevant to generating it.

Causal masking makes decoder prediction autoregressive. When predicting target position \(t\), the decoder may use earlier target positions but not the current or future target tokens. During training, all positions can still be processed in parallel because attention scores to future positions are replaced with a large negative value before softmax.

![Comparison of a lower-triangular causal attention mask during parallel training and autoregressive token-by-token inference](images/causal_mask_training_inference.png)
*Causal masking allows parallel training while enforcing one-token-at-a-time generation at inference.*

Finally, the top decoder representation is projected to the target vocabulary and converted into next-token probabilities:

\[
p(y_t\mid y_{<t},x)=\operatorname{softmax}(h_tW_{\text{out}}+b)
\]

The model selects or samples the next token, appends it to the target prefix, and repeats until it produces an end-of-sequence token.

## Understand the supporting components that make training work

Attention mixes information across positions, but the Transformer also relies on position-wise feed-forward networks, residual connections, normalization, teacher forcing, and masking.

A feed-forward network applies the same small multilayer perceptron independently to every token:

\[
\operatorname{FFN}(x)=\sigma(xW_1+b_1)W_2+b_2
\]

Residual connections add a sublayer’s output to its input:

\[
y = x + \operatorname{Sublayer}(x)
\]

Layer normalization is commonly represented as:

\[
\operatorname{LayerNorm}(x+\operatorname{Sublayer}(x))
\]

The decoder is trained with teacher forcing and a next-token prediction objective. The complete shifted target sequence is processed in parallel, while causal masking prevents future-token leakage. Inference is autoregressive: the model predicts one token, appends it, and predicts the next.

## Implement a minimal attention block and inspect its tensors

Scaled dot-product attention takes queries, keys, and values, computes query-key similarity, and uses the resulting weights to mix values:

```python
import math
import torch

def scaled_dot_product_attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(q.size(-1))

    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    weights = torch.softmax(scores, dim=-1)
    output = weights @ v
    return output, weights
```

The mask is applied before softmax. For causal attention, position \(i\) must not look at positions greater than \(i\):

```python
S = 4
causal_mask = torch.tril(torch.ones(S, S, dtype=torch.bool))
causal_mask = causal_mask[None, None, :, :]
```

In multi-head attention, projected tensors are reshaped from `(B, S, E)` to `(B, H, S, D)`, attention is computed independently for each head, and the results are transposed and recombined. During debugging, verify tensor shapes, that attention rows sum to approximately one, and that masked entries have near-zero weight.

## Close with the Transformer mental model and its trade-offs

A useful mental model is a repeated data path:

1. **Embed tokens and add positional information.**
2. **Mix information with attention.**
3. **Transform each position with a feed-forward network.**
4. **Stabilize each sublayer with residual connections and normalization.**
5. **Convert decoder representations into vocabulary probabilities.**

The central conceptual shift is that a token no longer has to receive information through a hidden state passed sequentially from earlier positions. Self-attention directly mixes information across a sequence, cross-attention connects decoder positions to encoder outputs, and multi-head attention learns several relationship patterns in parallel.

That flexibility has a cost. Full self-attention considers every pair of positions, so its attention matrix grows quadratically with sequence length. Longer inputs therefore increase memory use and computation rapidly, even though the operation is highly parallelizable.

When reviewing a Transformer, check positional information, query-key-value shapes, causal-mask orientation, head splitting and recombination, feed-forward and residual sublayers, normalization, padding masks, output logits, and the quadratic cost of full attention.

If you can trace those steps, you understand the Transformer’s core promise: replace sequential state passing with direct, learnable information mixing—while paying for that connectivity in attention cost.