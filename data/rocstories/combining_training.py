# combine_datasets.py

AUGMENTED_FILE = "data/rocstories/augmented_train.txt"
ORIGINAL_FILE = "data/rocstories/train.txt"
OUTPUT_FILE = "data/rocstories/final_train.txt"

START_INDEX = 32500  # 0-based index → line 32501

def load_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def main():
    print("Loading augmented data...")
    augmented = load_lines(AUGMENTED_FILE)

    print("Loading original train data...")
    original = load_lines(ORIGINAL_FILE)

    print(f"Total original stories: {len(original)}")

    # Take remaining stories
    remaining = original[START_INDEX:]

    print(f"Remaining stories taken: {len(remaining)}")

    # Combine
    final_data = augmented + remaining

    print(f"Final dataset size: {len(final_data)}")

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for story in final_data:
            f.write(story + "\n")

    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()