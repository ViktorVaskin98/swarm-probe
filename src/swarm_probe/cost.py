def generation_flops(n_params: int, prompt_tokens: int, generated_tokens: int) -> float:
    return 2.0 * n_params * (prompt_tokens + generated_tokens)
