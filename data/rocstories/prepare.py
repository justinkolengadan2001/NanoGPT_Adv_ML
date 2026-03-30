import os
import pickle
import re
import numpy as np
import tiktoken
from datasets import load_dataset

DATA_DIR = os.path.dirname(__file__)
enc = tiktoken.get_encoding("gpt2")

# -------------------------------------------------
# Basic dataset story extraction
# -------------------------------------------------

def normalize_story(example):
    # Preferred: dataset has a single text field
    if "text" in example and example["text"] is not None:
        return example["text"].strip()

    # Fallback: join sentence fields if present
    sent_keys = ["sentence1", "sentence2", "sentence3", "sentence4", "sentence5"]
    if all(k in example for k in sent_keys):
        return " ".join(example[k].strip() for k in sent_keys)

    raise KeyError(f"Could not find story text fields in example keys: {list(example.keys())}")


# -------------------------------------------------
# Cleaning helpers
# -------------------------------------------------

def normalize_unicode_punctuation(text):
    replacements = {
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2013": "-",   # en dash
        "\u2014": "-",   # em dash
        "\u2026": "...", # ellipsis
        "\xa0": " ",     # non-breaking space
    }
    for src, tgt in replacements.items():
        text = text.replace(src, tgt)
    return text


def remove_weird_symbols(text):
    # Keep letters, digits, whitespace, and common punctuation
    text = re.sub(r"[^A-Za-z0-9\s\.,!?;:'\"()\-\n]", " ", text)
    return text


def fix_punctuation_spacing(text):
    # Remove spaces before punctuation: "word ." -> "word."
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    # Ensure one space after punctuation where appropriate
    text = re.sub(r"([.,!?;:])([A-Za-z\"'])", r"\1 \2", text)

    return text


def normalize_exclamations(text, convert_to_period=True):
    # Collapse repeated punctuation
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)

    if convert_to_period:
        text = text.replace("!", ".")

    return text


def collapse_whitespace(text):
    # Replace multiple newlines with a single space
    text = re.sub(r"\n+", " ", text)

    # Collapse repeated spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def ensure_terminal_punctuation(text):
    text = text.strip()
    if not text:
        return text
    if text[-1] not in ".!?":
        text += "."
    return text


def basic_clean_story(text, convert_exclamation_to_period=True):
    text = text.strip()
    text = normalize_unicode_punctuation(text)
    text = remove_weird_symbols(text)
    text = normalize_exclamations(text, convert_to_period=convert_exclamation_to_period)
    text = fix_punctuation_spacing(text)
    text = collapse_whitespace(text)
    text = ensure_terminal_punctuation(text)
    return text


# -------------------------------------------------
# Sentence handling
# -------------------------------------------------

def split_into_sentences(text):
    text = text.strip()
    if not text:
        return []

    # Split after ., !, ? followed by whitespace
    parts = re.split(r'(?<=[.!?])\s+', text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def normalize_sentences(sentences, force_period=True):
    cleaned = []
    for s in sentences:
        s = re.sub(r"\s+", " ", s).strip()

        if force_period:
            s = re.sub(r"[!?]+$", ".", s)
            if s and s[-1] not in ".!?":
                s += "."
        else:
            if s and s[-1] not in ".!?":
                s += "."

        cleaned.append(s)

    return cleaned


def preprocess_story_to_five_sentences(
    story,
    strict_five=True,
    convert_exclamation_to_period=True,
):
    story = basic_clean_story(
        story,
        convert_exclamation_to_period=convert_exclamation_to_period
    )
    sentences = split_into_sentences(story)
    sentences = normalize_sentences(sentences, force_period=True)

    if strict_five:
        if len(sentences) != 5:
            return None
    else:
        if len(sentences) > 5:
            sentences = sentences[:5]

    return sentences


# -------------------------------------------------
# Optional genre loading
# -------------------------------------------------

def load_genres(filepath):
    if not os.path.exists(filepath):
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        genres = [line.strip() for line in f]

    return genres


# -------------------------------------------------
# Final formatting
# -------------------------------------------------

def format_story(sentences, genre=None, use_story_tag=True):
    parts = []

    if use_story_tag:
        parts.append("<|story|>")

    if genre is not None and genre.strip():
        parts.append(f"[Genre: {genre.strip()}]")

    parts.extend(sentences)
    return " ".join(parts).strip()

# def encode_split(stories, genres=None, verbose=True):
#     eot = enc.eot_token
#     ids = []
#     kept = 0
#     skipped = 0

#     for story in stories:
#         story = story.strip()
#         if not story:
#             skipped += 1
#             continue

#         sentences = preprocess_story_to_five_sentences(story, strict_five=True, convert_exclamation_to_period=True,)

#         if sentences is None or len(sentences) != 5:
#             skipped += 1
#             continue

#         # Plain format: sentence1 sentence2 sentence3 sentence4 sentence5 <|endoftext|>
#         text = " ".join(sentences).strip()

#         story_ids = enc.encode_ordinary(text)
#         ids.extend(story_ids)
#         ids.append(eot)

#         kept += 1

#     if verbose:
#         print(f"Kept stories   : {kept}")
#         print(f"Skipped stories: {skipped}")

#     return np.array(ids, dtype=np.uint16)

def encode_split(stories, genres=None, verbose=True):
    eot = enc.eot_token
    ids = []
    kept = 0
    skipped = 0

    # deterministic 60 / 20 / 20 split by story index
    # pattern of 5:
    # 0,1,2 -> plain story        (60%)
    # 3     -> <|story|> story    (20%)
    # 4     -> prompt/continuation (20%)

    for i, story in enumerate(stories):
        story = story.strip()
        if not story:
            skipped += 1
            continue

        sentences = preprocess_story_to_five_sentences(
            story,
            strict_five=True,
            convert_exclamation_to_period=True,
        )

        if sentences is None or len(sentences) != 5:
            skipped += 1
            continue

        plain_text = " ".join(sentences).strip()
        tagged_text = f"<|story|> {plain_text}".strip()
        prompt = sentences[0]
        continuation = " ".join(sentences[1:])
        prompt_cont_text = f"<|story|> Prompt: {prompt} Continuation: {continuation}".strip()

        mod = i % 5
        if mod in [0, 1, 2]:
            text = plain_text                 # 60%
        elif mod == 3:
            text = tagged_text               # 20%
        else:
            text = prompt_cont_text          # 20%

        story_ids = enc.encode_ordinary(text)
        ids.extend(story_ids)
        ids.append(eot)

        kept += 1

    if verbose:
        print(f"Kept stories   : {kept}")
        print(f"Skipped stories: {skipped}")

    return np.array(ids, dtype=np.uint16)

# Alternate version that includes all 3 formats for each story, which is more data but less variety
# def encode_split(stories, genres=None, verbose=True):
#     eot = enc.eot_token
#     ids = []
#     kept = 0
#     skipped = 0

#     for story in stories:
#         story = story.strip()
#         if not story:
#             skipped += 1
#             continue

#         sentences = preprocess_story_to_five_sentences(
#             story,
#             strict_five=True,
#             convert_exclamation_to_period=True,
#         )

#         if sentences is None or len(sentences) != 5:
#             skipped += 1
#             continue

#         plain_text = " ".join(sentences).strip()
#         tagged_text = f"<|story|> {plain_text}".strip()

#         prompt = sentences[0]
#         continuation = " ".join(sentences[1:]).strip()
#         prompt_cont_text = f"<|story|> Prompt: {prompt} Continuation: {continuation}".strip()

#         all_versions = [
#             plain_text,
#             plain_text,
#             tagged_text,
#             prompt_cont_text,
#         ]

#         for text in all_versions:
#             story_ids = enc.encode_ordinary(text)
#             ids.extend(story_ids)
#             ids.append(eot)

#         kept += 1

#     if verbose:
#         print(f"Kept original stories   : {kept}")
#         print(f"Skipped original stories: {skipped}")
#         print(f"Total sequences written : {kept * 3}")

#     return np.array(ids, dtype=np.uint16)

# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    ds = load_dataset("mintujupally/ROCStories")

    train_split = ds["train"]
    test_split = ds["test"]

    train_stories = [normalize_story(ex) for ex in train_split]
    test_stories = [normalize_story(ex) for ex in test_split]

    # assignment/train.py expects train.bin and val.bin
    train_ids = encode_split(train_stories, genres=None, verbose=True)
    val_ids = encode_split(test_stories, genres=None, verbose=True)

    print(f"train stories raw : {len(train_stories):,}")
    print(f"val stories raw   : {len(test_stories):,}")
    print(f"train tokens      : {len(train_ids):,}")
    print(f"val tokens        : {len(val_ids):,}")

    train_ids.tofile(os.path.join(DATA_DIR, "train.bin"))
    val_ids.tofile(os.path.join(DATA_DIR, "val.bin"))

    print("Saved train.bin, val.bin")

if __name__ == "__main__":
    main()