Write a GPU program that implements top-p (nucleus) sampling for LLM inference.

Top-p sampling is a text generation technique where you sample from the smallest set of tokens whose cumulative probability exceeds threshold p. This balances randomness and quality better than pure top-k or greedy sampling.

Given logits (unnormalized scores) from a language model:

1.  Convert logits to probabilities using softmax
2.  Sort tokens by probability (descending)
3.  Find the smallest set where cumulative probability ≥ p (the "nucleus")
4.  Renormalize the nucleus probabilities to sum to 1
5.  Sample a token from the nucleus using the provided random seed

先将 logits 转为概率并按概率降序排列，再找到累计概率达到阈值 `p` 的最小 token 集合，重新归一化后使用给定随机种子采样。

## Implementation Requirements / 实现要求

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- Ensure numerical stability when computing softmax

## Example 1 / 示例 1:

    Input:
      logits = [1.0, 2.0, 3.0, 0.5]
      p = 0.9
      seed = 42

    Output:
      sampled_token = 2 or 1
      (tokens with highest probabilities, sampled randomly)

## Example 2 / 示例 2:

    Input:
      logits = [10.0, 1.0, 1.0]
      p = 0.5
      seed = 123

    Output:
      sampled_token = 0
      (single token dominates the probability mass)

## Constraints / 约束

- 3 ≤ `vocab_size` ≤ 50,000
- -100.0 ≤ `logits[i]` ≤ 100.0
- 0.0 \< `p` ≤ 1.0
- 0 ≤ `sampled_token` \< vocab_size
- Performance is measured with `vocab_size` = 50,000
