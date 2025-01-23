"""Load a model and use it to generate text."""

import rootutils
import tiktoken
import torch as T
import torch.nn.functional as F
from tqdm import trange

root = rootutils.setup_root(search_from=__file__, pythonpath=True)


from src.models.gpt import GPT
from src.models.mpgpt import MPGPT

T.set_grad_enabled(False)


def main():
    # Load the tokenizer.
    enc = tiktoken.get_encoding("gpt2")

    # Load the model from the lightning checkpoint.
    model_path = "/srv/beegfs/scratch/groups/rodem/nlp/gpt/MP-GPT/checkpoints/last.ckpt"
    model = MPGPT.load_from_checkpoint(model_path)
    model.eval()
    model.requires_grad_(False)

    # Get the maximum input tokens
    max_len = 1024
    max_new_tokens = 100
    temp = 0.9
    top_k = None
    text = "The Mystery of the Hidden Garden. Leon Bozianu was no ordinary boy; he was a young adventurer with an insatiable curiosity. Living in the quaint village of St. Eloi, Leon spent his days exploring the picturesque landscape that surrounded his home. His favorite spot was the old, abandoned mansion at the edge of town, a place filled with secrets and wonders. One sunny afternoon, as Leon was wandering through the overgrown garden of the mansion, he stumbled upon an ancient key buried beneath a bed of ivy. The key was intricate, with delicate engravings and a hint of rust. Intrigued, Leon slipped the key into his pocket, determined to uncover its secrets. One sunny afternoon, as Leon was wandering through the overgrown garden of the mansion, he stumbled upon an ancient key buried beneath a bed of ivy. The key was intricate, with delicate engravings and a hint of rust. Intrigued, Leon slipped the key into his pocket, determined to uncover its secrets."

    # Do a loop to generate the new tokens
    while True:
        new = input("Enter the next text: ")
        if new == "restart":
            text = ""
            continue
        text += new
        print(text)
        tokens = enc.encode_ordinary(text)

        # Convert the input to pytorch tensor.
        tokens = T.tensor(tokens, dtype=T.long, device=model.device).unsqueeze(0)

        for _ in trange(max_new_tokens):
            # Trim the input to the maximum sequence length.
            input_tokens = (
                tokens if tokens.shape[1] <= max_len else tokens[:, -max_len:]
            )

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

        print("-------------------------")
        print(text)
        print("-------------------------")


if __name__ == "__main__":
    main()
