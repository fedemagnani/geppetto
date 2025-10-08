- The goal of training a LLM is to **maximize the likelihood of the correct token**, given the previous context tokens.
- Training is performed via **backpropagation**, tipycally used in deep neural networks.
  -  It requires a loss function to evaluate how good the model is performing
- **Cross entropy** is a technique to measue how different are two probability distributions. It is used (and optimized) during training of LLMs.
- **Perplexity** is another metric: it proxies the "uncertainty" of a model when predicting the next token. You can tke the exponential of the cross-entropy to get the perplexity. The output is a value **k** and you can interpret saying that the model is as uncertain as if it had to pick between **k** different options uniformly. A lower perplexity is better, becasue model is more sure about its predictions.

- Adam optimizers are typically used to train LLMs
  - AdamW is a variant of Adam that improves the weight decay approach, which aims to minimize model complexity and prevent overfitting by penalizing larger weights (enforces a regularization to favor stability)

- If a model is trained on a small dataset over multiple epochs, it would more likely overfit the training data, leading to poor generalization on unseen data. 

-  When a vector of logits is returned by the model, we need to decide which entry to pick, because the associated index will be the predicted token id.
   -  **Greedy decoding**: pick the entry associated to the highest logit -> text will be repetitive and not so diverse
   -  Map the logits to a probability distribution via softmax, and then sample from it, assuming that the probabilities follow a **multinomial distribution** 
      -  They call **temperature scaling** the dowscaling of logits before applying softmax, by a certain constant called temperature. So **each logit is divided by the temperature value**. A very high temperature value will make the softmax output more uniform (giving more "variety" to the text, because it nerfs the probability of the most likely token), while a very low temperature will make the softmax output peakier.
   -  **Top-k sampling**. Select the top k logits, mapping all the other logits to -infinity (so that their softmax probability will be 0), and then sample from the resulting distribution: this is equivalent to a multinomial distribution with a way more restricted support. You can perform temperature scaling to make distribution of the top-k logits more or less uniform.