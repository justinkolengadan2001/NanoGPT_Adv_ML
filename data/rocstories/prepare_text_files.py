import os
import re
import numpy as np
import tiktoken

DATA_DIR = os.path.dirname(__file__)
enc = tiktoken.get_encoding("gpt2")


# -------------------------------------------------
# Reading plain text files
# -------------------------------------------------

def load_stories_from_txt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    # one story per line
    return [line for line in lines if line]


# -------------------------------------------------
# Cleaning helpers
# -------------------------------------------------

def normalize_unicode_punctuation(text):
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\xa0": " ",
    }
    for src, tgt in replacements.items():
        text = text.replace(src, tgt)
    return text

def remove_weird_symbols(text):
    return re.sub(r"[^A-Za-z0-9\s\.,!?;:'\"()\-\n]", " ", text)


def fix_punctuation_spacing(text):
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([.,!?;:])([A-Za-z\"'])", r"\1 \2", text)
    return text


def normalize_exclamations(text, convert_to_period=True):
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    if convert_to_period:
        text = text.replace("!", ".")
    return text


def collapse_whitespace(text):
    text = re.sub(r"\n+", " ", text)
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
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


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
# Plain encoding only
# -------------------------------------------------

def encode_split(stories, verbose=True):
    eot = enc.eot_token
    ids = []
    kept = 0
    skipped = 0

    for story in stories:
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

        # Plain format only:
        # sentence1 sentence2 sentence3 sentence4 sentence5 <|endoftext|>
        text = " ".join(sentences).strip()

        story_ids = enc.encode_ordinary(text)
        ids.extend(story_ids)
        ids.append(eot)

        kept += 1

    if verbose:
        print(f"Kept stories   : {kept}")
        print(f"Skipped stories: {skipped}")

    return np.array(ids, dtype=np.uint16)


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    train_path = os.path.join(DATA_DIR, "final_train.txt")
    val_path = os.path.join(DATA_DIR, "test.txt")

    train_stories = load_stories_from_txt(train_path)
    val_stories = load_stories_from_txt(val_path)

    train_ids = encode_split(train_stories, verbose=True)
    val_ids = encode_split(val_stories, verbose=True)

    print(f"train stories raw : {len(train_stories):,}")
    print(f"val stories raw   : {len(val_stories):,}")
    print(f"train tokens      : {len(train_ids):,}")
    print(f"val tokens        : {len(val_ids):,}")

    train_ids.tofile(os.path.join(DATA_DIR, "train.bin"))
    val_ids.tofile(os.path.join(DATA_DIR, "val.bin"))

    print("Saved train.bin, val.bin")


if __name__ == "__main__":
    main()