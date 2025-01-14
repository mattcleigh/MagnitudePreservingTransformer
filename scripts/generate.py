"""Load a model and use it to generate text."""

import rootutils
import tiktoken
import torch as T
import torch.nn.functional as F


root = rootutils.setup_root(search_from=__file__, pythonpath=True)


from src.models.gpt import GPT

T.set_grad_enabled(False)

# Load the tokenizer.
enc = tiktoken.get_encoding("gpt2")

# Load the model from the lightning checkpoint.
model_path = "/srv/beegfs/scratch/groups/rodem/nlp/gpt/all_rms/checkpoints/last.ckpt"
model = GPT.load_from_checkpoint(model_path)
model.eval()
model.requires_grad_(False)

# Get the maximum input tokens
max_len = model.max_seq_len
max_new_tokens = 100
temp = 1
top_k = 20
text = ""

# Do a loop to generate the new tokens
while True:
    text += input("Enter the text to continue: ")
    tokens = enc.encode_ordinary(text)

    # Convert the input to pytorch tensor.
    tokens = T.tensor(tokens, dtype=T.long, device=model.device).unsqueeze(0)

    for _ in range(max_new_tokens):
        # Trim the input to the maximum sequence length.
        input_tokens = tokens if tokens.shape[1] <= max_len else tokens[:, -max_len:]

        # Pass through the model (by default it will only return the final logit)
        logits, _ = model(input_tokens)

        # Crop the logits to only the top k options
        if top_k is not None:
            v, _ = T.topk(logits, k=min(top_k, logits.shape[-1]))
            min_k = v[0, 0, -1]
            logits[logits < min_k] = -float("Inf")

        probs = F.softmax(logits / temp, dim=-1)
        idx = T.multinomial(probs[-1], num_samples=1)

        # Append to the text and tokens
        tokens = T.hstack([tokens, idx])
        text += enc.decode(idx[0].tolist())

    print(text)
