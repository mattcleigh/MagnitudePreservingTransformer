"""Download, tokenize, and save the WikiText-103 dataset."""

import numpy as np
import rootutils
import tiktoken
from datasets import load_dataset

root = rootutils.setup_root(search_from=__file__)


def main():
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    enc = tiktoken.get_encoding("gpt2")

    def prepare(x):
        return {"ids": enc.encode_ordinary(x["text"])}

    ds = ds.map(prepare, remove_columns=["text"], num_proc=4)

    # Loop over the splits, save as a single numpy array (only 1GB for train)
    for split, data in ds.items():
        print(f"Saving {split} split")
        file_name = root / "data" / "wikitext-103" / f"{split}.npy"
        file_name.parent.mkdir(parents=True, exist_ok=True)
        arr = np.hstack(data.with_format("numpy")["ids"])
        np.save(file_name, arr.astype(np.uint16))


if __name__ == "__main__":
    main()
