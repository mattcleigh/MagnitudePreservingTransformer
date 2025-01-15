from pathlib import Path

import numpy as np
import rootutils
import tiktoken
from datasets import load_dataset
from tqdm import trange

root = rootutils.setup_root(search_from=__file__)

N_THREADS = 10


def main():
    cache_path = Path("/home/users/l/leighm/scratch/cache")
    save_path = Path("/home/users/l/leighm/scratch/openwebtext")
    cache_path.mkdir(parents=True, exist_ok=True)
    save_path.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("openwebtext", num_proc=N_THREADS, cache_dir=cache_path)
    ds = ds["train"].train_test_split(test_size=0.0005, seed=42, shuffle=True)

    def prepare(x):
        ids = enc.encode_ordinary(x["text"])
        ids.append(enc.eot_token)  # Tells the model that the segment has ended
        return {"ids": ids, "len": len(ids)}

    ds = ds.map(prepare, remove_columns=["text"], num_proc=N_THREADS)

    # Loop over the splits, save as a single numpy array (only 1GB for train)
    for split, data in ds.items():
        print(f"Saving {split} split")

        # Calculate the total length of the split
        total_len = np.sum(data["len"], dtype=np.uint64)

        # Initialise the binary file
        file_name = save_path / f"{split}.bin"
        arr = np.memmap(file_name, dtype=np.uint16, mode="w+", shape=(total_len,))

        # Fill in the binary file in chunks
        n_chunks = 1000
        idx = 0
        for chunk_idx in trange(n_chunks):
            chunk = data.shard(num_shards=n_chunks, index=chunk_idx, contiguous=True)
            chunk = np.hstack(chunk.with_format("numpy")["ids"])
            arr[idx : idx + len(chunk)] = chunk
            idx += len(chunk)
        arr.flush()


if __name__ == "__main__":
    main()
