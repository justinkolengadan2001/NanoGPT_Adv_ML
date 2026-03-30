import re
import subprocess
from pathlib import Path
from typing import List, Optional

# =========================
# CONFIG
# =========================
INPUT_FILE = "train.txt"
OUTPUT_FILE = "genre_train.txt"
FAILED_FILE = "genre_failed.txt"
PROGRESS_FILE = "genre_progress.txt"

OLLAMA_MODEL = "llama3.2:3b"      # change to llama3.1:8b or llama3.2:3b if wanted
TIMEOUT_SECONDS = 60
MAX_RETRIES = 3

# Chunking
START_INDEX = 0
END_INDEX = None   # set to an int for partial run

# Resume
RESUME = True

VALID_LABELS = {"Humorous", "Heartwarming", "Suspenseful", "Sad", "Neutral"}

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
Classify the overall tone of this 5-sentence story into exactly one of these labels:

Humorous
Heartwarming
Suspenseful
Sad
Neutral

Rules:
- Output exactly one label only.
- Do not explain.
- Do not output anything else.
- Choose the single best label.

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

def extract_label(text: str) -> Optional[str]:
    text = text.strip()

    # Remove simple code fences if present
    text = re.sub(r"^```[a-zA-Z0-9_]*\n", "", text)
    text = re.sub(r"\n```$", "", text)
    text = text.strip()

    # Exact match first
    if text in VALID_LABELS:
        return text

    # Otherwise find first valid label mentioned
    for label in VALID_LABELS:
        if re.search(rf"\b{re.escape(label)}\b", text, flags=re.IGNORECASE):
            return label

    return None

# =========================
# VALIDATION
# =========================

def validate_label(label: Optional[str]) -> bool:
    return label in VALID_LABELS

# =========================
# CLASSIFICATION
# =========================

def classify_story(story: str, model: str, max_retries: int = MAX_RETRIES) -> Optional[str]:
    story = clean_story(story)

    if not is_valid_five_sentence_story(story):
        return "Neutral"

    for _ in range(max_retries):
        try:
            raw = call_ollama(build_prompt(story), model)
            label = extract_label(raw)

            if validate_label(label):
                return label

        except Exception:
            continue

    return None

# =========================
# MAIN
# =========================

def main():
    stories = load_stories(INPUT_FILE)

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
            story = clean_story(story)

            if not is_valid_five_sentence_story(story):
                append_line(OUTPUT_FILE, "Neutral")
                append_line(FAILED_FILE, story)
                write_progress(global_idx + 1)
                print("  -> invalid story, using Neutral", flush=True)
                continue

            label = classify_story(story, OLLAMA_MODEL)

            if label is None:
                append_line(OUTPUT_FILE, "Neutral")
                append_line(FAILED_FILE, story)
                write_progress(global_idx + 1)
                print("  -> failed, using Neutral", flush=True)
                continue

            append_line(OUTPUT_FILE, label)
            write_progress(global_idx + 1)
            print(f"  -> {label}", flush=True)

        except KeyboardInterrupt:
            print("\nStopped by user. Progress saved.", flush=True)
            write_progress(global_idx)
            raise

        except Exception as e:
            append_line(OUTPUT_FILE, "Neutral")
            append_line(FAILED_FILE, story)
            write_progress(global_idx + 1)
            print(f"  -> error: {e}", flush=True)

    print("\nDone.", flush=True)
    print(f"Output saved to: {OUTPUT_FILE}", flush=True)
    print(f"Failures saved to: {FAILED_FILE}", flush=True)
    print(f"Progress saved in: {PROGRESS_FILE}", flush=True)

if __name__ == "__main__":
    main()