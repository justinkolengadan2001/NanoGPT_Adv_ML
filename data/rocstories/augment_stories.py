import re
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

# =========================
# CONFIG
# =========================
INPUT_FILE = "train.txt"
OUTPUT_FILE = "augmented_train.txt"
FAILED_FILE = "failed_augments.txt"
PROGRESS_FILE = "augment_progress.txt"

OLLAMA_MODEL = "llama3.2:3b"      # change to llama3.1:8b or llama3.2:3b if wanted
TIMEOUT_SECONDS = 60
MAX_RETRIES = 3

# Chunking
START_INDEX = 7500 # 7500
END_INDEX = 32500 # 32500   # set None for all remaining stories

# Resume
RESUME = True

# =========================
# TEXT UTILITIES
# =========================

def normalize_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def split_into_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    sentences = re.findall(r'[^.!?]+[.!?]', text)
    return [s.strip() for s in sentences]

def is_valid_five_sentence_story(text: str) -> bool:
    return len(split_into_sentences(text)) == 5

def clean_story(text: str) -> str:
    text = normalize_text(text)
    sents = split_into_sentences(text)
    if len(sents) == 5:
        return " ".join(sents)
    return text

# =========================
# FILE IO
# =========================

def load_stories(path: str) -> List[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [normalize_text(x) for x in lines if x.strip()]

def append_line(path: str, line: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.strip() + "\n")

def write_progress(idx: int) -> None:
    Path(PROGRESS_FILE).write_text(str(idx), encoding="utf-8")

def read_progress() -> int:
    p = Path(PROGRESS_FILE)
    if p.exists():
        txt = p.read_text(encoding="utf-8").strip()
        if txt.isdigit():
            return int(txt)
    return START_INDEX

# =========================
# PROMPT
# =========================

def build_prompt(story: str) -> str:
    return f"""
You are helping create augmented ROCStories-style training data.

Given this 5-sentence story:
- Keep sentence 1 EXACTLY unchanged.
- Rewrite only sentences 2 to 5 in two ways:
  1. positive ending
  2. negative ending

Rules:
- Each output must be exactly 5 sentences.
- Sentence 1 must be copied exactly.
- Use simple natural everyday language.
- Keep the story realistic and ROCStories-like.
- Do not add commentary.
- Do not use markdown.
- Output in exactly this format:

POSITIVE: full 5-sentence story
NEGATIVE: full 5-sentence story

Story:
{story}
""".strip()

# =========================
# OLLAMA
# =========================

def call_ollama(prompt: str, model: str) -> str:
    result = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=TIMEOUT_SECONDS
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()

def extract_labeled_output(text: str) -> dict:
    text = text.strip()

    # Remove simple code fences if present
    text = re.sub(r"^```[a-zA-Z0-9_]*\n", "", text)
    text = re.sub(r"\n```$", "", text)

    pos_match = re.search(r"POSITIVE:\s*(.+?)(?=\nNEGATIVE:|\Z)", text, flags=re.DOTALL)
    neg_match = re.search(r"NEGATIVE:\s*(.+)", text, flags=re.DOTALL)

    if not pos_match or not neg_match:
        raise ValueError("Could not parse POSITIVE/NEGATIVE output.")

    return {
        "positive": pos_match.group(1).strip(),
        "negative": neg_match.group(1).strip(),
    }

# =========================
# VALIDATION
# =========================

def validate_augmented_story(original_s1: str, story: str) -> bool:
    story = clean_story(story)
    sents = split_into_sentences(story)

    if len(sents) != 5:
        return False

    if sents[0].strip() != original_s1.strip():
        return False

    return True

# =========================
# AUGMENTATION
# =========================

def augment_story(story: str, model: str, max_retries: int = MAX_RETRIES) -> Optional[Tuple[str, str]]:
    story = clean_story(story)
    sents = split_into_sentences(story)

    if len(sents) != 5:
        return None

    original_s1 = sents[0]

    for _ in range(max_retries):
        try:
            raw = call_ollama(build_prompt(story), model)
            data = extract_labeled_output(raw)

            positive = clean_story(data["positive"])
            negative = clean_story(data["negative"])

            if not validate_augmented_story(original_s1, positive):
                continue
            if not validate_augmented_story(original_s1, negative):
                continue

            return positive, negative

        except Exception:
            continue

    return None

# =========================
# MAIN
# =========================

def main():
    stories = load_stories(INPUT_FILE)

    # Light input filtering
    stories = [clean_story(s) for s in stories if is_valid_five_sentence_story(s)]

    chunk_start = START_INDEX
    if RESUME:
        chunk_start = max(chunk_start, read_progress())

    chunk_end = END_INDEX if END_INDEX is not None else len(stories)
    stories_chunk = stories[chunk_start:chunk_end]

    total = len(stories_chunk)
    print(f"Processing stories from index {chunk_start} to {chunk_end} (count={total})", flush=True)

    for local_idx, story in enumerate(stories_chunk, start=1):
        global_idx = chunk_start + local_idx - 1
        print(f"[{local_idx}/{total}] Global index {global_idx}...", flush=True)

        try:
            result = augment_story(story, OLLAMA_MODEL)

            if result is None:
                append_line(FAILED_FILE, story)
                write_progress(global_idx + 1)
                print("  -> failed", flush=True)
                continue

            positive, negative = result

            append_line(OUTPUT_FILE, story)
            append_line(OUTPUT_FILE, positive)
            append_line(OUTPUT_FILE, negative)

            write_progress(global_idx + 1)
            print("  -> success", flush=True)

        except KeyboardInterrupt:
            print("\nStopped by user. Progress saved.", flush=True)
            write_progress(global_idx)
            raise

        except Exception as e:
            append_line(FAILED_FILE, story)
            write_progress(global_idx + 1)
            print(f"  -> error: {e}", flush=True)

    print("\nDone.", flush=True)
    print(f"Output saved to: {OUTPUT_FILE}", flush=True)
    print(f"Failures saved to: {FAILED_FILE}", flush=True)
    print(f"Progress saved in: {PROGRESS_FILE}", flush=True)

if __name__ == "__main__":
    main()