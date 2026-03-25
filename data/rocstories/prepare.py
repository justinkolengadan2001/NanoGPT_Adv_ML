import os
import pickle
import numpy as np
import tiktoken
from datasets import load_dataset

DATA_DIR = os.path.dirname(__file__)
enc = tiktoken.get_encoding("gpt2")

def normalize_story(example):
    # Preferred: dataset has a single text field
    if "text" in example and example["text"] is not None:
        return example["text"].strip()

    # Fallback: join sentence fields if present
    sent_keys = ["sentence1", "sentence2", "sentence3", "sentence4", "sentence5"]
    if all(k in example for k in sent_keys):
        return " ".join(example[k].strip() for k in sent_keys)

    raise KeyError(f"Could not find story text fields in example keys: {list(example.keys())}")

# def encode_split(stories):
#     # Join stories with blank lines, as instructed
#     # text = "\n\n".join(stories)
#     # ids = enc.encode_ordinary(text)
#     # return np.array(ids, dtype=np.uint16)

#     eot = enc.eot_token  # end-of-text token
#     ids = []

#     for story in stories:
#         story = story.strip()
#         story_ids = enc.encode_ordinary(story)
#         ids.extend(story_ids)
#         ids.append(eot)  

#     return np.array(ids, dtype=np.uint16)

def encode_split(stories):
    eot = enc.eot_token
    ids = []

    for story in stories:
        story = story.strip()
        story_ids = enc.encode_ordinary(story)
        ids.extend(story_ids)
        ids.extend(enc.encode_ordinary("\n\n"))  # optional separator
        ids.append(eot)

    return np.array(ids, dtype=np.uint16)

def main():
    ds = load_dataset("mintujupally/ROCStories")

    print(ds)

    train_split = ds["train"]
    test_split = ds["test"]

    train_stories = [normalize_story(ex) for ex in train_split]
    test_stories = [normalize_story(ex) for ex in test_split]

    # assignment/train.py expects train.bin and val.bin
    train_ids = encode_split(train_stories)
    val_ids = encode_split(test_stories)

    print(f"train stories: {len(train_stories):,}")
    print(f"val stories  : {len(test_stories):,}")
    print(f"train tokens : {len(train_ids):,}")
    print(f"val tokens   : {len(val_ids):,}")

    train_ids.tofile(os.path.join(DATA_DIR, "train.bin"))
    val_ids.tofile(os.path.join(DATA_DIR, "val.bin"))

    meta = {
        "vocab_size": enc.n_vocab
    }
    with open(os.path.join(DATA_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print("Saved train.bin, val.bin, meta.pkl")

if __name__ == "__main__":
    main()