- Problem: **langiage translation**
  - Translating is a difficult problem because word-by-word translation is faulty
  - Different languages have different grammar and structure, a word-by-word translation would make you sound like Yoda

To overcome this issue, deep neural networks introduced two submodules:
  - **Encoder**: read and process the entire text
  - **Decoder**: generates the translation

Before 2017, the **RNNs were the most popular encoder-decoder arhcitecture**:
    - The encoder used to process each word, **updateing each time the internal hidden state**, trying to capture the meaning of the entire sentence
    - The encoder's final hidden state was then evaluated by the decoder to start generating the translated text. At each step, the decoder internal state is updated as well, to carry context necessary to predict the next word to be generated.
    - **Main problem**: earlier encoder hidden state is not accessible: this can lead to loss of context, especially for long sentences
    - **This shortcoming motivated the design of attention mechanisms**

"Self"-attention means the capacity of learning relationship and dependencies between parts of the input itself, such as words in a sentence or pixels in a image:
    - For each embedding (i.e. vector associated to a token-id), it is computed a **context vector**, which combines information from all other input elements.
      - You can think about the context vector as an **enriched embedding vector**
      - This representation is "enriched" because it incorporates information from all other elements in the sequence.
    - The contribution of each input element to the context vector of another element is measured by the **attention-weights** (normalization of the attention-scores)
      - This means that each vector-embedding doesn't have a sigle attention-weight, but rather multiple attention-weights, one for each other input element
      - For example, in order to compute the **context vector** of vector embedding 2, you combine (vector_embedding_1, a_21), (vector_embedding_2, a_22), (vector_embedding_3, a_23), ... where a_2i is the attention weight of vector embedding i to compute the context vector of vector embedding 2
      - The attention weights are learned during training
      - The **attention scores** for a certain vector embedding "i" is a vector whose entries are the **inner product** between the vector embedding "i" and all other vector embeddings (including itself) : attention_scores_i = [<x_i, x_1>, <x_i, x_2>, <x_i, x_3>, ...]
      - Each entry of the attention score measures the similiarity between the vector embedding "i" and the other vector embeddings: the higher the similiarity, the higher is the attention score between the two elements.
      - The **attention scores are then normalized, in order to come up with weights summing to one**, in this way you obtain the **attention weights**: to perform this normalization, you can 
        - divide eache entry by the sum of all entries (you project on the L1 ball)
        - using the **softmax** function: more advisable to handle extreme values and offer better numerical stability
    - The **contextx vector** is then computed by summing each vector embedding weightd by the corresponding attention weight: context_vector_i = sum_j (attention_weight_ij * vector_embedding_j). 
      - This means that to compute the context vector of vector embedding "i", you iterate over each entry of the attention weight associated with "i and multiply the j-th entry wirh the j-th vector embedding, summing all these contributions
    - So, suppose that you have **k inital vector embeddings**, this procedure will output **k context vectors**, which can be thought as the initial vector embeddings enriched by contextual information
    
- Given that each row of the input matrix is the vector embedding of the i-th subword in the sentence, and given that the attention score vector is the vector of inner products between the i-th vector embedding and all other vector embeddings, it follows naturally that you can compute attention scores via matrix multiplication **input @ input.T**
  - Each row of the resultin matrix will correspond to the attention score vector of the i-th vector embedding
  - Applying the softmax function at each row, you **turn the i-th row into the attention weight vector of the i-th vector embedding**
    - You can check that, iterating over each row, the entries sum to one
  - Then, in order to make sure that the j-th entry of the i-th attention weight si multiplied by the j-th vector embedding and then summed, to compute the context vector you have **context_vector = attention_weights @ inputs**: each row of the resulting matrix will be the context vector associated with the i-th vector embedding
  - As you can see, the **context vector is simply the projection of input embeddings under a linear transformation (vector of weights) which enriches the input vector embeddings with contextual information**

- The methodology described so far is a **simplified self-attention mechanism** to compute the context vector: there isn't any learnable parameter in this methodology, as the attention scores are completely characterized by the input vector embeddings and the attention weights are simply the normalization of the attention scores.
- Normal self-attention systems rely instead on **three trainable weight matrices**, each computing a **"special" context vector**. Notice that these vectors **may or may not have the same dimension as the input vector embeddings**. Notice that the following matrices **are not attention weights**, but wwill be involved in the computation of the attention weights:
  - **W_q** -> projects input vector embeddings into context vectors called **query vectors**. This matrix enriches the inpput vector embeddings with some contextual information
  - **W_k** -> projects input vector embeddings into context vectors called **key vectors**. This matrix enriches the inpput vector embeddings with some contextual information
  - **W_v** -> projects input vector embeddings into context vectors called **value vectors**. This matrix enriches the inpput vector embeddings with some contextual information
- Recap: starting from a single subword tokenized by a tokenizer, you have the following transformations chain:
  - subword -> token_id -> vector_embedding -> (query_vector, key_vector, value_vector)
  - Notice how each subword "explodes" in dimensionality, and it might even refer to just a part of a word!
- In order to compute the attention score **a_{ij}** we take the **dot product** between the **i-th query vector** and the **j-th key vector**: a_{ij} = <q_i, k_j>
  - Before, we were computing the attention scores just by taking the inner product of the input vector embeddings, while here we are computing the attention scores based on different projections of the input vector embeddings
  - You can think about the simplified model as a special case of this more general model, where W_q = W_k = identity matrix (constant, not trainable)
  - As a result, the matrix of attention scores (each row is the attention score vector of the i-th input vector embedding) is computed as **attention_scores = query_matrix @ keys_matrix.T**
  - Then, in order to obtain the attention weights, you apply the softmax function at each row of the attention scores matrix. Typically, you divide the attention scores by sqrt(d_k) before applying the softmax function, **where d_k is the dimension of the key vectors**: this adjustements avoids gradient vanishing/explosion problems when d_k is very large
    - This scaling is also the reason why this attention mechanism is called **"scaled dot-product attention"**
  - Finally, the context vectors are computed as **context_vectors = attention_weights @ value_matrix**
    - As before, the trivial self-attention mechanis is like computing the context vectors setting the weight values **W_v = identity matrix**

- **Causal/Masked attention** is an attention mechanism based on computing attention scores just by looking to previous inputs.
- The goal is to come up with a lower-triangular matrix of attention weights
  - The "lazy" way is computing attention scores as usual, then extracting the lower-diagonal part (masking) and normalize again iterating on each row (renormalization)
  - Fortunately, this combination of masking + renormalization saves from information leakage, so is absolutely viable

- **Multi-Headed attention**: In multi-headed attention you have multiple sets of (W_q, W_k, W_v) matrices
  - each set (W_q, W_k, W_v) is called a **"head"**. Each "head" computes its own context vectors and is trained independently.
  - The attention mechanism employed in each head can be causal (the matrix of attention weights will be masked) or not
  - The attention mechanism is run in parralel (one thread per head), 
  - We know that each head produces a matrix of context vectors (one context vector per input vector embedding), so a multi-headed attention with "h" heads produces a stack of **"h" context matrices**. These matrices are horizontally concatenated, so that if each context matrix has `d_out` columns, the resulting concatenated matrix will have `h * d_out` columns. The number of rows remain the same, equal to the number of input vector embeddings