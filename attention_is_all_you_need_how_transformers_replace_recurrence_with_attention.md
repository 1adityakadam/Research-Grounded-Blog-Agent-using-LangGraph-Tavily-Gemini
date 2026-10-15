# Attention Is All You Need: How Transformers Replace Recurrence with Attention

## Motivate the Transformer architecture

A sequence model must combine information across positions: a word may depend on the token immediately before it, on a phrase several sentences earlier, or on both at once. Recurrent neural networks (RNNs) process tokens step by step. At position \(t\), the model updates a hidden state using the current token and the state from position \(t-1\). This gives recurrence a natural notion of order, but it also creates two challenges. Computation is inherently sequential, limiting parallelism during training, and information from distant positions must pass through many state updates. Even with carefully designed gates, signals can weaken or become difficult to preserve across long sequences.

Convolutional models offer more parallel computation: a filter can process many positions simultaneously. However, each layer sees only a local window. Connecting distant tokens therefore requires stacking multiple layers or using larger and more expensive filters. Both recurrent and convolutional approaches make relationships between far-apart positions indirect.

This leads to the central design question: **can every token directly incorporate information from any other token in the sequence?** If so, a representation at one position could selectively use nearby context, distant context, or both, without waiting for information to travel step by step.

The Transformer answers this question with attention as its primary mechanism. Rather than relying on recurrence or convolution to mix sequence information, it combines attention with position information, position-wise feed-forward layers, normalization, and residual connections. This separation lets attention handle communication between tokens while the other components transform, stabilize, and preserve the resulting representations.

## Build intuition for self-attention

Self-attention lets every token look at every other token in the same sequence and decide which information is useful. A helpful analogy is an information-retrieval system:

- A **query** describes what the current token is looking for.
- A **key** describes what kind of information each token contains.
- A **value** is the information that will actually be retrieved if a key matches the query.

Suppose a sequence contains token representations \(x_1, x_2, \ldots, x_n\), where \(x_i\) is the vector for token \(i\). For one token, the Transformer creates three new vectors by learned linear projections:

\[
q_i = x_i W_Q,\qquad k_i = x_i W_K,\qquad v_i = x_i W_V
\]

Here, \(q_i\), \(k_i\), and \(v_i\) are the query, key, and value for token \(i\). The matrices \(W_Q\), \(W_K\), and \(W_V\) contain learned parameters. They let the same input representation play three different roles: asking a question, advertising what it contains, and supplying content.

For token \(i\), self-attention compares its query \(q_i\) with every key \(k_j\) in the sequence. A common similarity score is the dot product:

\[
s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}}
\]

The index \(j\) identifies a candidate source token, and \(d_k\) is the number of components in each key and query vector. Dividing by \(\sqrt{d_k}\) keeps scores from becoming excessively large as vector dimensions grow.

The raw scores are converted into attention weights with softmax:

\[
a_{ij} = \operatorname{softmax}_j(s_{ij})
\]

For a fixed token \(i\), these weights are nonnegative and sum to one. A high \(a_{ij}\) means that token \(i\) considers token \(j\)'s information especially relevant.

Finally, token \(i\) combines the value vectors using those weights:

\[
z_i = \sum_{j=1}^{n} a_{ij}v_j
\]

The result \(z_i\) is a context-aware representation. It still corresponds to token \(i\), but now incorporates information from the entire sequence.

> **[IMAGE GENERATION FAILED]** Self-attention routes information by matching each query with every key, then combining the corresponding values.
>
> **Alt:** Diagram showing token representations projected into queries, keys, and values; query-key scores become softmax weights that mix value vectors into contextualized token outputs
>
> **Prompt:** Create a clean technical educational diagram for a machine-learning explainer titled “Self-attention: from tokens to context”. Show a horizontal row of input token vectors on the left, branching through three clearly labeled learned projection boxes: Query Q, Key K, and Value V. Show one highlighted query comparing against all keys, producing a row of similarity scores, then a softmax attention-weight row, then a weighted sum over value vectors producing one contextualized output token. Use arrows, compact matrix/vector symbols, short labels, and consistent blue/purple/green colors. White background, precise transformer architecture style, no decorative elements, no paragraphs, no tiny unreadable text.
>
> **Error:** 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-3.1-flash-lite-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-3.1-flash-lite-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-3.1-flash-lite-image\nPlease retry in 7h19m35.578933305s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_input_token_count', 'quotaId': 'GenerateContentInputTokensPerModelPerMinute-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-3.1-flash-lite-image'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-3.1-flash-lite-image'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-3.1-flash-lite-image'}}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '26375s'}]}}


For example, in “The animal crossed the street because it was tired,” the token **it** can assign strong attention to **animal**, even though other words intervene. Its updated representation can therefore encode that **it** likely refers to the animal rather than merely relying on nearby word order.

## Derive Scaled Dot-Product Attention

Attention begins with three matrices: **queries**, **keys**, and **values**. Suppose an input sequence contains \(L\) token positions. For each position, a learned projection produces:

- \(Q \in \mathbb{R}^{L_q \times d_k}\): queries, representing what each position is looking for
- \(K \in \mathbb{R}^{L_k \times d_k}\): keys, representing what each position offers for matching
- \(V \in \mathbb{R}^{L_k \times d_v}\): values, representing the information to retrieve

Here, \(d_k\) is the query/key feature width, and \(d_v\) is the value feature width. In self-attention, all three come from the same sequence, so \(L_q=L_k=L\), although their learned projections are different.

The scaled dot-product attention equation is:

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\mathsf{T}}{\sqrt{d_k}}\right)V
\]

The calculation proceeds in three main steps.

First, transpose the key matrix:

\[
K^\mathsf{T} \in \mathbb{R}^{d_k \times L_k}
\]

Then multiply it by \(Q\):

\[
QK^\mathsf{T}
\in
\mathbb{R}^{L_q \times L_k}
\]

Each element in this score matrix is a dot product between one query and one key. Thus, row \(i\) contains the compatibility scores between query position \(i\) and every key position.

The scores are divided by \(\sqrt{d_k}\). Without this scaling, dot products tend to grow in magnitude as the feature dimension increases. Large positive or negative values make softmax distributions extremely peaked, causing most weights to approach zero and producing weak gradients during training. Dividing by the square root of the dimension keeps score magnitudes in a more useful range.

Next, softmax is applied independently across each row:

\[
A
=
\operatorname{softmax}\left(\frac{QK^\mathsf{T}}{\sqrt{d_k}}\right)
\in
\mathbb{R}^{L_q \times L_k}
\]

Every row of \(A\) sums to one. Its entries are therefore mixing weights: for a particular query position, they specify how much to take from each value position.

Finally, multiply these weights by \(V\):

\[
AV
\in
\mathbb{R}^{L_q \times d_v}
\]

The output has one \(d_v\)-dimensional representation for each query position. Shape-checking the operation gives:

\[
(L_q \times L_k)(L_k \times d_v)
=
L_q \times d_v
\]

The flexibility of attention comes with a cost. Because the score matrix relates every query position to every key position, it contains \(L_qL_k\) pairwise interactions. For self-attention, this becomes \(L^2\), producing quadratic sequence-length memory and computation for the attention matrix.

## Multi-Head Attention: Learning Several Relationship Patterns

Multi-head attention applies self-attention several times in parallel, using different learned representation spaces. Let the input sequence be \(X \in \mathbb{R}^{n \times d_{\text{model}}}\), where \(n\) is the number of tokens and \(d_{\text{model}}\) is the width of each token representation.

For head \(i\), the model learns three projection matrices:

\[
Q_i = XW_i^Q,\qquad K_i = XW_i^K,\qquad V_i = XW_i^V
\]

Here, \(Q_i\), \(K_i\), and \(V_i\) are the queries, keys, and values for that head. Although every head receives the same \(X\), its learned matrices project the tokens into a different, usually lower-dimensional, space.

Each head then independently performs scaled dot-product attention:

\[
\text{head}_i =
\operatorname{softmax}\left(\frac{Q_iK_i^\top}{\sqrt{d_k}}\right)V_i
\]

The matrix \(Q_iK_i^\top\) contains a score for every query-token and key-token pair. Dividing by \(\sqrt{d_k}\), where \(d_k\) is the key dimension, keeps these scores at a useful scale before the softmax converts them into weights. The result is a weighted mixture of value vectors for each token.

After all heads compute their outputs, the model concatenates them along the feature dimension and applies one more learned projection:

\[
\operatorname{MHA}(X)
=
\operatorname{Concat}(\text{head}_1,\ldots,\text{head}_h)W^O
\]

Different heads can specialize in different relationships. One may emphasize nearby words and local syntax, another may connect a pronoun to its antecedent for coreference, and another may attend according to positional or long-range structural patterns. These are learned tendencies, not manually assigned rules.

A single large attention operation could use the same total number of features, but it would provide one shared similarity space. Separate heads create multiple subspaces, allowing the model to inspect different relationship types simultaneously before combining their evidence. Thus, multi-head attention is not a different kind of attention; it is parallel self-attention with deliberately separated projections.

## Assemble the Transformer encoder and decoder

A Transformer is not just an attention operation. It is an architecture that stacks several attention and feed-forward layers, connects them with residual paths, and uses normalization to keep information and gradients well behaved.

Let \(X \in \mathbb{R}^{n \times d}\) represent a sequence of \(n\) token vectors, each with model width \(d\). An **encoder layer** contains two sublayers:

1. **Multi-head self-attention**, where every input position can attend to every other input position.
2. A **position-wise feed-forward network**, applied independently to each position.

The feed-forward network is typically a small multilayer perceptron:

\[
\operatorname{FFN}(x) = W_2\,\sigma(W_1x+b_1)+b_2
\]

Here, \(W_1\) and \(W_2\) are learned matrices, \(b_1\) and \(b_2\) are biases, and \(\sigma\) is a nonlinear activation. “Position-wise” means the same network is reused for every token, without mixing positions; attention performs the mixing.

Each sublayer is wrapped with a **residual connection** and **layer normalization**. In one common arrangement:

\[
H' = \operatorname{LayerNorm}(H + \operatorname{Attention}(H))
\]

\[
H_{\text{next}} = \operatorname{LayerNorm}(H' + \operatorname{FFN}(H'))
\]

The residual addition gives information a short path through the network, making it easier for a layer to learn a refinement rather than an entirely new representation. Layer normalization rescales and recenters activations, helping optimization remain stable as many layers are stacked.

The **decoder** also contains repeated layers, but each layer begins with **masked multi-head self-attention**. During generation, the prediction at position \(i\) may use earlier target tokens, but must not inspect tokens at positions \(j>i\). A causal mask enforces this rule by preventing attention to future positions. Without it, training could reveal the correct next token directly, creating information that will not be available when generating autoregressively.

A decoder layer then performs **encoder-decoder cross-attention**. The decoder’s current representations produce the queries \(Q\), while the encoder outputs provide the keys \(K\) and values \(V\):

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

Thus, each decoder position can select relevant information from the encoded input sequence. This is distinct from self-attention: self-attention uses one sequence for queries, keys, and values, while cross-attention connects the decoder to the encoder.

Attention alone does not know whether one token came before or after another. Before entering the encoder, token embeddings therefore receive **positional information**, such as a learned positional embedding or a deterministic positional encoding. The resulting vectors carry both token identity and location.

End to end, source tokens become embeddings plus positional information, pass through the encoder stack, and produce contextual representations. The decoder consumes previously generated target tokens, adds their positional information, applies masked self-attention and cross-attention, and produces a final hidden vector at each position. A linear projection followed by softmax converts that vector into probabilities over the output vocabulary. Generation selects or samples a token, feeds it back into the decoder, and repeats until an end-of-sequence token is produced.

> **[IMAGE GENERATION FAILED]** The Transformer separates source encoding from autoregressive target generation, linked by encoder-decoder cross-attention.
>
> **Alt:** Transformer encoder-decoder architecture showing positional embeddings, encoder self-attention, decoder masked self-attention, cross-attention, feed-forward layers, and output probabilities
>
> **Prompt:** Create a precise technical block diagram of the original Transformer encoder-decoder architecture. On the left, show source tokens flowing through token embeddings plus positional encoding, then a vertical stack of encoder layers, each containing multi-head self-attention, add and layer normalization, feed-forward network, and another add and normalization. On the right, show shifted target tokens plus positional encoding flowing through decoder layers containing masked multi-head self-attention, add and normalization, encoder-decoder cross-attention receiving keys and values from the encoder, add and normalization, and feed-forward network. End with linear projection and softmax output probabilities. Clearly distinguish self-attention, masked self-attention, and cross-attention with labeled arrows. Use a clean white background, compact labels, aligned boxes, and no decorative imagery.
>
> **Error:** 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-3.1-flash-lite-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-3.1-flash-lite-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-3.1-flash-lite-image\nPlease retry in 7h19m35.392826996s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-3.1-flash-lite-image'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-3.1-flash-lite-image'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_input_token_count', 'quotaId': 'GenerateContentInputTokensPerModelPerMinute-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-3.1-flash-lite-image'}}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '26375s'}]}}


## Implement a Minimal Attention Layer

Assume a batch contains \(B\) sequences, each with \(T\) token representations of width \(D\). Let

- \(X \in \mathbb{R}^{B \times T \times D}\) be the input token representations.
- \(W_Q, W_K \in \mathbb{R}^{D \times D_k}\) project inputs into queries and keys.
- \(W_V \in \mathbb{R}^{D \times D_v}\) projects inputs into values.
- \(Q = XW_Q\), \(K = XW_K\), and \(V = XW_V\).

For self-attention, all three projections come from the same \(X\). The attention score between token \(i\)'s query and token \(j\)'s key is their dot product. Dividing by \(\sqrt{D_k}\) keeps these scores from becoming excessively large as the key width grows.

The complete operation is:

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{D_k}}\right)V
\]

Here, \(QK^\top\) has shape \(B \times T \times T\): for every sequence, it contains one score for every query-token/key-token pair.

```python
import math
import torch
import torch.nn as nn

class MinimalSelfAttention(nn.Module):
    def __init__(self, d_model, d_k=None, d_v=None):
        super().__init__()
        d_k = d_k or d_model
        d_v = d_v or d_model

        self.query = nn.Linear(d_model, d_k, bias=False)
        self.key   = nn.Linear(d_model, d_k, bias=False)
        self.value = nn.Linear(d_model, d_v, bias=False)

    def forward(self, x, mask=None):
        # x: [B, T, D]
        q = self.query(x)  # [B, T, D_k]
        k = self.key(x)    # [B, T, D_k]
        v = self.value(x)  # [B, T, D_v]

        # Transpose the final two dimensions of k:
        # [B, T, D_k] @ [B, D_k, T] -> [B, T, T]
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(q.size(-1))

        if mask is not None:
            # mask should broadcast to [B, T, T].
            # True means "allowed"; False means "blocked".
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=-1)  # [B, T, T]
        output = weights @ v                     # [B, T, D_v]
        return output, weights
```

A **padding mask** prevents attention from using placeholder positions added to make sequences the same length. It usually blocks key columns corresponding to padding tokens; implementations may also suppress outputs at padded query positions.

A **causal mask** enforces autoregressive order. For position \(i\), it allows attention only to positions \(j \leq i\), blocking future tokens. A lower-triangular mask implements this rule:

```python
T = x.size(1)
causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
```

This example deliberately omits multi-head splitting, an output projection, dropout, residual connections, layer normalization, feed-forward layers, and positional information. Production systems also use optimized attention kernels, careful memory management, supported mask formats, and more sophisticated batching. Those pieces belong to a full Transformer block or architecture; the code above isolates the core self-attention mechanism.

## Strengths, Limitations, and Design Trade-offs

The central benefit of attention is **direct interaction**. In self-attention, each token can compare its representation with every other token in the same sequence in a single layer. Information does not need to travel step by step through intermediate recurrent states. This makes long-range relationships easier to represent and allows the training computation for many tokens to run in parallel on modern hardware.

That flexibility has a fundamental cost. Let \(n\) denote the sequence length. Standard self-attention forms an interaction score for every pair of positions, producing an \(n \times n\) score matrix. Its computation and memory therefore grow approximately as \(O(n^2)\). Doubling the sequence length can require roughly four times as many pairwise scores. The projection operations also consume resources, but the quadratic interaction pattern is the main constraint for long inputs.

Attention also does not inherently know whether one token comes before or after another. Positional information must be supplied explicitly, such as by adding positional vectors to token representations, or through an architectural mechanism that makes positions distinguishable. Without some positional signal, a sequence can look too much like an unordered collection of tokens.

Parallelism mainly describes **training**. During training, a model can process all positions in a known input sequence together, subject to its attention mask. Autoregressive inference is different: when generating text from left to right, token \(t+1\) depends on the tokens already generated, so each new token must generally be produced sequentially. Cached key and value representations reduce repeated work, but they do not remove this dependency.

Attention weights can provide useful diagnostic evidence about which positions interact, yet they should not automatically be treated as explanations of a model’s decision. They are internal routing values, and different mechanisms can sometimes produce similar outputs without assigning importance in the same way.

These trade-offs support several common architectural patterns. **Encoder-only** models use bidirectional context for tasks such as representation and classification. **Decoder-only** models use causal masking for left-to-right generation. **Encoder-decoder** models combine an input-processing encoder with a generating decoder, allowing the decoder to attend to the encoded input. These are design choices built around the same attention machinery, not fundamentally different kinds of computation.

> **[IMAGE GENERATION FAILED]** Self-attention considers an all-pairs interaction matrix; causal masking removes future-token entries while retaining the quadratic layout.
>
> **Alt:** Comparison of full self-attention and causal attention matrices, illustrating all-to-all interaction, triangular masking, and quadratic sequence-length growth
>
> **Prompt:** Create a compact technical comparison diagram with three panels. Panel 1: a square n by n self-attention matrix with every cell active, labeled “bidirectional self-attention”. Panel 2: a lower-triangular matrix with the upper triangle visibly blocked, labeled “causal mask: token i attends only to j ≤ i”. Panel 3: a simple scaling visual showing sequence length n increasing and the number of pairwise scores growing as n squared, labeled “O(n²)”. Use an intuitive heatmap/grid style, clear axes labeled query position and key position, restrained blue and gray colors, white background, concise readable labels, no decorative elements.
>
> **Error:** 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-3.1-flash-lite-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-3.1-flash-lite-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-3.1-flash-lite-image\nPlease retry in 7h19m35.172935781s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_input_token_count', 'quotaId': 'GenerateContentInputTokensPerModelPerMinute-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-3.1-flash-lite-image'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-3.1-flash-lite-image'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'model': 'gemini-3.1-flash-lite-image', 'location': 'global'}}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '26375s'}]}}


## Conclude with a practical mental model

A Transformer layer can be read as a repeated information-processing pipeline:

1. **Represent tokens** with embeddings, then add position information so the model can distinguish order.
2. **Mix information with attention**: each position compares its query with other positions’ keys and uses the resulting weights to combine their values.
3. **Transform each position** with a feed-forward network, usually applied independently after attention has mixed context.
4. **Preserve and stabilize computation** with residual paths and normalization around these sublayers.
5. **Produce predictions** by applying an output projection to the final hidden representations.

When reading or designing a model, ask:

- Is attention computed between positions in the same sequence? If so, it is **self-attention**.
- Does one sequence attend to representations from another sequence? That is **cross-attention**.
- Can a position see future tokens? A **causal mask** should prevent those attention scores from contributing.
- Are there several independently parameterized attention operations whose outputs are combined? That is **multi-head attention**.
- Are attention, feed-forward, residual, and normalization blocks arranged into an encoder, decoder, or both? That describes the **full architecture**, not merely the attention mechanism.

The original Transformer’s deeper idea is to replace fixed information paths—such as recurrence passing state step by step—with learned, content-dependent connections. A token can select which other tokens matter for the current computation, while the surrounding layers refine and preserve that information.

For practice, visualize an attention matrix and interpret each row as a position’s information-gathering pattern. Then modify a minimal attention layer to apply a causal mask and observe how the matrix changes.

The central insight is simple: **attention lets every position selectively gather information from other positions.**