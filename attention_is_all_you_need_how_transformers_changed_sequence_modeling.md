# Attention Is All You Need: How Transformers Changed Sequence Modeling

## Start with the sequence-modeling problem

Many language tasks can be framed as **sequence-to-sequence** problems: a model receives one sequence and produces another. In translation, for example, the input is a sentence in one language and the output is its equivalent in another. The model must understand both individual tokens and how tokens relate across the entire sequence.

Before Transformers, recurrent neural networks were a common solution. A recurrent model reads tokens one at a time, maintaining a hidden state that summarizes everything seen so far:

```text
state = initial_state
for token in input_sequence:
    state = update(state, token)
```

This ordering is useful for modeling time and word order, but it creates two limitations. First, tokens cannot be processed independently, which limits parallelism during training. Second, information from an early token must pass through many recurrent updates before influencing a later token. Although mechanisms such as gated recurrent units and long short-term memory help preserve information, long-range dependencies can still be difficult to learn and represent.

Convolutional sequence models process multiple positions at once, making them more parallelizable. However, a convolution typically combines only nearby positions. To connect distant tokens, the network needs a larger receptive field: more layers, wider kernels, or repeated operations. Thus, a relationship between two far-apart words may require information to travel through several intermediate layers.

The Transformer takes a different approach. Instead of passing information only along a chain or through local neighborhoods, it lets each position directly consider other positions using **attention**. For a given token, the model computes how relevant every other token is and forms a weighted combination of their representations. This exposes long-range relationships in a single layer and allows many positions to be processed in parallel.

The original Transformer uses an **encoder-decoder** design. The encoder reads the input sequence and builds contextual representations for all its positions. The decoder generates the output sequence, using its previously generated tokens and information from the encoder. Later sections will distinguish the attention mechanisms inside these components and show how they replace the recurrence that earlier sequence models depended on.

## Build Intuition for Scaled Dot-Product Attention

Attention lets each token build a context-aware representation by combining information from other tokens. Instead of processing every token with only its local neighbors, a token can assign different importance to all tokens in the sequence.

For each token, the model creates three learned representations through separate linear projections:

- **Query**: what information this token is looking for.
- **Key**: what information this token offers for matching.
- **Value**: the content to contribute if the token is considered relevant.

Suppose a sequence contains \(n\) tokens. Let \(Q\), \(K\), and \(V\) be matrices containing the queries, keys, and values for those tokens. If each query and key has dimension \(d_k\), and each value has dimension \(d_v\), scaled dot-product attention is

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\mathsf{T}}{\sqrt{d_k}}\right)V.
\]

The matrix \(QK^\mathsf{T}\) contains one score for every query-key pair. A score at row \(i\), column \(j\) measures how compatible token \(i\)'s query is with token \(j\)'s key. The dot product is a natural compatibility measure: aligned vectors produce a large positive value, while unrelated or opposing directions produce a smaller value.

Each row of the score matrix is divided by \(\sqrt{d_k}\), the square root of the key dimension. Without this scaling, dot products tend to grow in magnitude as the vector dimension increases. Large scores make softmax distributions extremely sharp: one position may receive almost all the probability, while gradients for other positions become very small. Scaling keeps the softmax inputs in a more manageable range, allowing attention to remain selective without becoming prematurely brittle.

Softmax converts each row of scores into nonnegative weights that sum to one. Multiplying those weights by \(V\) produces one output vector per query. In implementation terms, the operation is conceptually:

```text
scores  = Q @ transpose(K)
scores  = scores / sqrt(key_dimension)
weights = rowwise_softmax(scores)
output  = weights @ V
```

> **[IMAGE GENERATION FAILED]** Scaled dot-product attention compares queries with keys, normalizes the scores, and uses the resulting weights to mix value vectors.
>
> **Alt:** Diagram showing query, key, and value projections flowing into scaled dot-product attention and a context output
>
> **Prompt:** Create a clean technical diagram for an educational blog about Transformers. Show a short token sequence entering three parallel learned linear projections labeled Query (what each token seeks), Key (what each token offers for matching), and Value (content to retrieve). Show Q and transposed K producing an n-by-n score matrix, division by sqrt(d_k), row-wise softmax producing attention weights, and multiplication by V producing one context vector per query. Use concise labels, arrows, matrix-like blocks, and a restrained blue and charcoal palette. No decorative characters, no photorealism, no excessive text, high legibility.
>
> **Error:** 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-3.1-flash-lite-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-3.1-flash-lite-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-3.1-flash-lite-image\nPlease retry in 7h18m21.033987793s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_input_token_count', 'quotaId': 'GenerateContentInputTokensPerModelPerMinute-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-3.1-flash-lite-image'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-3.1-flash-lite-image'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-3.1-flash-lite-image'}}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '26301s'}]}}


Consider the sentence, “The animal crossed the street because it was tired.” In a bidirectional self-attention layer, the query for **“it”** may be especially compatible with the key for **“animal”**. Its attention weights could therefore emphasize that token, helping “it” form a representation connected to its likely antecedent. A token can similarly attend to a relevant later token when the attention pattern permits looking ahead.

Attention does not normally select exactly one token. The output for “it” might combine information from **“animal,” “crossed,” “street,”** and other positions, with different weights for each. It is therefore best understood as a learned, context-dependent weighted mixture of value vectors—not as a single hard lookup.

## Multi-Head Attention: Learning in Multiple Representation Subspaces

Multi-head attention lets the model examine the same sequence through several learned representation subspaces. Let the input sequence be \(X \in \mathbb{R}^{n \times d_{\text{model}}}\), where \(n\) is the number of tokens and \(d_{\text{model}}\) is the model’s representation width.

For each head \(i\), the Transformer learns separate projection matrices:

\[
Q_i = XW_i^Q,\qquad K_i = XW_i^K,\qquad V_i = XW_i^V
\]

Here, \(Q_i\), \(K_i\), and \(V_i\) are the queries, keys, and values for head \(i\). Each projection maps the same input into a different learned subspace. A query represents what a token is looking for; a key represents what another token offers for matching; and a value contains the information to retrieve.

A head computes scaled dot-product attention:

\[
\text{Attention}(Q_i,K_i,V_i)
=
\text{softmax}\left(\frac{Q_iK_i^\top}{\sqrt{d_h}}\right)V_i
\]

The symbol \(d_h\) is the dimensionality of one head. The dot product \(Q_iK_i^\top\) produces a score for every query-key pair. Dividing by \(\sqrt{d_h}\) keeps these scores at a useful scale, while softmax converts them into weights. The weighted sum of values becomes the head’s output.

The important idea is that different heads can learn different relationships. One head may assign high weight to nearby tokens or useful positional patterns. Another may connect a verb with its subject, while another associates a pronoun with a relevant noun or links semantically related words. These are not manually assigned roles: training discovers projections that help the overall prediction objective.

The sequence of operations is:

1. Project the input into queries, keys, and values for each head.
2. Split or reshape the results so the head dimension is explicit.
3. Compute attention independently in every head.
4. Concatenate the head outputs.
5. Apply a final learned projection \(W^O\) to mix information across heads.

Conceptually, a forward pass looks like this:

```text
function multi_head_attention(X):
    Q = project(X, W_Q)  # shape: [batch, tokens, heads, head_dim]
    K = project(X, W_K)
    V = project(X, W_V)

    Q, K, V = transpose_to_head_major(Q, K, V)

    scores = (Q @ transpose(K)) / sqrt(head_dim)
    weights = softmax(scores, axis=last_dimension)
    heads = weights @ V

    combined = concatenate_heads(heads)
    return combined @ W_O
```

Although there are multiple heads, they do not multiply the model’s total representation size without limit. Usually, \(d_h = d_{\text{model}} / h\), where \(h\) is the number of heads. Concatenating \(h\) outputs therefore returns to approximately \(d_{\text{model}}\) dimensions before the final projection. More heads provide more parallel subspaces, but each individual head becomes narrower and computation and parameters must still be managed.

Finally, head visualizations are useful diagnostics, not definitive explanations. A head may appear to track syntax or position in some examples without implementing one simple, human-interpretable function everywhere. Interpretation requires examining behavior across inputs and layers, not assigning a meaning from a single attention pattern.

## Assemble the Transformer encoder and decoder

A Transformer sequence-to-sequence model has two stacks: an **encoder** that reads the source sequence and a **decoder** that generates the target sequence. Let \(X \in \mathbb{R}^{n \times d}\) denote a sequence of \(n\) token representations, each with model width \(d\). Token embeddings are combined with positional encodings before entering the first layer, so the model receives both content and order information.

### Encoder layers

An encoder layer contains two sublayers:

1. Multi-head self-attention
2. A position-wise feed-forward network

In the original post-normalization formulation, each sublayer uses a residual connection followed by layer normalization:

\[
H' = \operatorname{LayerNorm}(H + \operatorname{SelfAttention}(H))
\]

\[
H_{\text{out}} =
\operatorname{LayerNorm}(H' + \operatorname{FFN}(H'))
\]

Here, \(H\) is the layer input and \(H_{\text{out}}\) is its output. The residual path lets information bypass a sublayer, making optimization easier and preserving earlier representations while each sublayer learns a refinement.

Self-attention allows every source position to compare itself with every other source position. Thus, a token representation can incorporate context from the entire input sentence, not only nearby positions.

### Decoder layers and masked self-attention

A decoder layer adds a third sublayer. Its order is:

1. Masked multi-head self-attention
2. Encoder-decoder attention
3. Position-wise feed-forward network

Decoder self-attention is **causally masked**. At target position \(t\), attention may use positions \(1\) through \(t\), but not positions greater than \(t\). During training, the complete target sentence is available, so the mask prevents the model from cheating by reading future target tokens.

The decoder input is normally the target sequence shifted right: a start symbol begins the input, while the expected next tokens serve as labels. This makes parallel training possible while preserving the autoregressive prediction task.

### Encoder-decoder attention

After masked self-attention, the decoder attends to the encoder output. In attention notation, \(Q\) represents queries, \(K\) keys, and \(V\) values. For encoder-decoder attention:

- \(Q\) comes from the decoder’s current representations.
- \(K\) and \(V\) come from the final encoder representations.

The decoder therefore asks source-dependent questions—such as which input positions help predict the next target token—and retrieves information from the encoded source. This differs from self-attention, where queries, keys, and values originate from the same sequence.

### Position-wise feed-forward network

The feed-forward network applies the same nonlinear transformation independently to every position:

\[
\operatorname{FFN}(x) = W_2\,\sigma(W_1x + b_1) + b_2
\]

The intermediate dimension is typically larger than \(d\). Attention mixes information *between* positions; the feed-forward network transforms each position’s features *within* that position. The activation \(\sigma\) supplies nonlinearity.

### Stacking and end-to-end flow

Multiple encoder and decoder layers are stacked. Lower layers can form local or syntactic features, while later layers combine increasingly contextual and task-relevant information.

A simplified flow is:

```text
source tokens
  -> embeddings + positions
  -> encoder layer × N
  -> encoded source representations

shifted target tokens
  -> embeddings + positions
  -> masked self-attention
  -> encoder-decoder attention over encoder output
  -> feed-forward network
  -> decoder layer × N
  -> linear projection + softmax
  -> next-token probabilities
```

> **[IMAGE GENERATION FAILED]** The encoder builds source representations; the decoder combines causal self-attention with cross-attention to generate target tokens.
>
> **Alt:** Encoder-decoder Transformer architecture showing source encoding, masked decoder self-attention, cross-attention, and next-token probabilities
>
> **Prompt:** Create a precise technical architecture diagram of the original Transformer encoder-decoder. On the left, show source tokens flowing through embeddings plus positional information and stacked encoder layers containing multi-head self-attention, residual plus normalization, and feed-forward blocks. On the right, show shifted target tokens flowing through masked self-attention, then encoder-decoder cross-attention receiving keys and values from the encoder output, then feed-forward blocks, linear projection, and next-token probabilities. Clearly label Q from the decoder and K/V from the encoder in the cross-attention connection. Include a small downward triangular causal mask icon in decoder self-attention. Use a clean academic infographic style, aligned boxes, arrows, blue for encoder, orange for decoder, white background, minimal text.
>
> **Error:** 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-3.1-flash-lite-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-3.1-flash-lite-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-3.1-flash-lite-image\nPlease retry in 7h18m20.860997762s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_input_token_count', 'quotaId': 'GenerateContentInputTokensPerModelPerMinute-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-3.1-flash-lite-image'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaDimensions': {'model': 'gemini-3.1-flash-lite-image', 'location': 'global'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-3.1-flash-lite-image'}}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '26300s'}]}}


During generation, the decoder emits one token, appends it to its prefix, and predicts the next token repeatedly. Thus, training evaluates many target positions in parallel with a causal mask, while inference proceeds autoregressively one position at a time.

## Handle Order with Positional Information

Self-attention compares every token with every other token, but the operation itself does not inherently know where a token appears. If we represent a sequence as token vectors \(x_1, x_2, \ldots, x_n\), attention can compute relationships among those vectors without using the indices \(1, 2, \ldots, n\). Consequently, rearranging the same inputs can produce the same collection of attention inputs. Without additional position information, the sequence risks being treated like an unordered set.

Transformers address this by adding a position-dependent vector to each token embedding. Let \(e_i\) be the embedding for the token at position \(i\), and let \(p_i\) be the positional vector for that position. The input to the first Transformer layer is

\[
z_i = e_i + p_i.
\]

The attention mechanism then operates on \(z_i\), so token content and location are available together. This addition does not change the attention algorithm; it changes the representations that attention receives.

The original Transformer used fixed sinusoidal positional encodings. For model dimension \(d\), position \(i\), and feature index \(k\), the encoding is

\[
p_{i,2k} = \sin\left(\frac{i}{10000^{2k/d}}\right),
\qquad
p_{i,2k+1} = \cos\left(\frac{i}{10000^{2k/d}}\right).
\]

Here, \(p_{i,2k}\) and \(p_{i,2k+1}\) are paired dimensions. Different pairs oscillate at different frequencies: some change quickly across positions, while others change slowly. Together, these values give each position a distinctive pattern and provide smooth signals from which relationships between positions can be inferred.

A different design is to learn the positional vectors. Instead of calculating \(p_i\), the model stores one trainable vector for each supported position and updates it during training. Learned positional embeddings leave the attention mechanism unchanged, but allow the model to discover position patterns that suit its data. Fixed encodings and learned embeddings both supply the missing location signal.

Useful positional information must distinguish positions and help represent both absolute relationships—“this is position 3”—and relative relationships—“this token is two places after that one.”

For example, “dog bites man” and “man bites dog” contain the same word embeddings. Without position signals, their inputs are the same collection of vectors. With positional vectors added, “dog” receives a different representation when it is first than when it is third, allowing attention and later layers to model the different meanings.

## Separate Training-Time Parallelism from Inference-Time Generation

Transformers are trained in parallel, but an autoregressive decoder still generates one token at a time. The difference comes from how the target sequence is supplied.

Let the target tokens be \(y_1, y_2, \ldots, y_T\). During training, **teacher forcing** gives the decoder the shifted sequence:

\[
\text{input: } \langle bos\rangle, y_1, y_2, \ldots, y_{T-1}
\]

The desired outputs are:

\[
\text{target: } y_1, y_2, \ldots, y_T
\]

Thus, the input at position \(t\) is the correct previous token, rather than a token predicted by the model. This makes it possible to present all \(T\) positions to the decoder in one tensor operation.

That parallelism must not allow a position to inspect its answer. A **causal mask** prevents position \(t\) from attending to positions greater than \(t\). If \(S\) is the attention-score matrix, where \(S_{t,u}\) measures how much position \(t\) attends to position \(u\), the mask sets \(S_{t,u}=-\infty\) whenever \(u>t\). After softmax, those entries receive probability zero:

```text
scores = query @ key.T / sqrt(head_dim)
scores[t, u] = -infinity  if u > t
weights = softmax(scores, axis="keys")
```

Every target position can therefore be evaluated in parallel while seeing only earlier inputs.

Inference has a different dependency pattern. The model first receives the beginning-of-sequence token, predicts a distribution for \(y_1\), then feeds the selected token back to predict \(y_2\), and continues until an end token appears:

```text
tokens = [BOS]
while not finished:
    next_token = decoder(tokens).last_position.argmax()
    tokens.append(next_token)
```

The training loss is usually **cross-entropy** summed or averaged over target positions. If \(p_t(k)\) is the predicted probability of vocabulary token \(k\) at position \(t\), the loss for the correct token \(y_t\) is:

\[
\mathcal{L}=-\sum_{t=1}^{T}\log p_t(y_t)
\]

Batches contain sequences of different lengths, so shorter examples are padded. A padding mask prevents padded keys from affecting attention and excludes padded target positions from this loss. Causal and padding masks solve different problems: one enforces temporal order; the other identifies nonexistent sequence elements.

Full self-attention compares every query position with every key position. For sequence length \(T\) and hidden width \(d\), this requires \(O(T^2)\) attention-score storage and roughly \(O(T^2d)\) score computation. Long sequences therefore increase both memory use and runtime rapidly.

Practical implementations must keep tensor shapes explicit, commonly \((\text{batch}, \text{heads}, \text{sequence}, \text{head\_dim})\), and broadcast masks to compatible dimensions. During inference, cached keys and values from earlier decoder steps avoid recomputing their projections. The output is still sequential, but each step reuses the decoder states already calculated.

> **[IMAGE GENERATION FAILED]** Teacher forcing evaluates all target positions in parallel during training, while inference generates one token at a time and can cache earlier keys and values.
>
> **Alt:** Side-by-side comparison of parallel teacher-forced training with a causal mask and sequential autoregressive inference
>
> **Prompt:** Create a side-by-side technical comparison diagram titled Training versus Inference for an autoregressive Transformer decoder. Left panel: show shifted target input BOS, y1, y2, y3 entering the decoder simultaneously, a lower-triangular causal attention mask blocking future positions, and parallel outputs y1, y2, y3, y4. Right panel: show BOS entering first, then a loop where the predicted token is appended to the prefix and the next token is generated sequentially; annotate optional cached K/V reuse. Use clear arrows, compact labels, a visible triangular mask matrix, and a restrained blue/orange palette. Educational vector infographic, no decorative imagery, high legibility.
>
> **Error:** 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-3.1-flash-lite-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-3.1-flash-lite-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-3.1-flash-lite-image\nPlease retry in 7h18m20.639876455s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-3.1-flash-lite-image'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-3.1-flash-lite-image'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_input_token_count', 'quotaId': 'GenerateContentInputTokensPerModelPerMinute-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-3.1-flash-lite-image'}}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '26300s'}]}}


## Close with a Practical Mental Model and Implementation Checklist

A useful implementation-level picture is:

1. Convert token IDs into embeddings and add positional information.
2. Pass the resulting sequence through repeated attention and feed-forward blocks.
3. Project each final hidden state into vocabulary-sized logits, then apply softmax when probabilities are needed.

For a tensor-shape checklist, let:

- \(B\) = batch size
- \(L\) = sequence length
- \(d_{\text{model}}\) = model/embedding dimension
- \(H\) = number of attention heads
- \(d_{\text{head}} = d_{\text{model}}/H\) = dimension per head
- \(V\) = vocabulary size

The input representation has shape \((B, L, d_{\text{model}})\). After linear projections, queries, keys, and values usually retain that shape. Reshaping for multi-head attention gives \((B, H, L, d_{\text{head}})\). Attention scores have shape \((B, H, L, L)\): every query position compares with every key position. The block output returns to \((B, L, d_{\text{model}})\), and the vocabulary projection produces \((B, L, V)\).

A compact attention sketch is:

```text
Q = X @ Wq                         # (B, H, L, d_head)
K = X @ Wk
V = X @ Wv

scores = (Q @ transpose(K)) / sqrt(d_head)  # (B, H, L, L)
scores = mask_future_positions(scores)      # decoder self-attention
weights = softmax(scores, axis=key_position)
context = weights @ V                        # (B, H, L, d_head)
output = merge_heads(context) @ Wo           # (B, L, d_model)
```

When debugging, watch for conceptual errors as well as shape errors. Keys are learned representations of positions, not token IDs. Decoder self-attention requires a causal mask; without it, a position can access future tokens during training. Encoder outputs are the sequence representations consumed by decoder cross-attention, whereas decoder states are autoregressive representations being updated. Finally, softmax must normalize over the keys being selected—typically the last \(L\) axis—not over batches, heads, or feature dimensions.

A productive learning sequence is incremental: implement single-head attention, add a causal mask, extend it to multiple heads, and compare each intermediate tensor and final output with a trusted implementation. Test tiny, hand-checkable inputs before optimizing.

The core Transformer mechanism is attention, position information, feed-forward transformation, residual connections, and normalization. Later choices—pretraining objectives, tokenizer design, layer layouts, and other architectural modifications—shape particular systems but are separate from that core.