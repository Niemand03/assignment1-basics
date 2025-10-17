import regex as re
import collections
from tqdm import tqdm
from pathlib import Path
import time

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe_tokenizer_naive(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    assert vocab_size >= 256 + len(special_tokens)
    
    vocab = {i: bytes([(i)]) for i in range(256)} # vocabulary initialization
    next_id = 256
    
    # --- add special tokens to vocabulary ---
    for s in special_tokens:
        vocab[next_id] = s.encode("utf-8")
        next_id += 1

    # --- pre-tokenization (word frequencys) ---
    word_freqs = collections.defaultdict(int)
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    if special_tokens:
        special_pattern = "|".join(re.escape(s) for s in special_tokens)
        text_chunks = re.split(f'({special_pattern})', text)
    else:
        text_chunks = [text]

    for chunk in text_chunks:
        if chunk in special_tokens:
            continue
        for match in re.finditer(PAT, chunk):
            word_bytes = match.group(0).encode("utf-8")
            word_freqs[tuple(word_bytes)] += 1

    # --- bpe merges ---
    num_bpe_merges_max = vocab_size - len(vocab)
    merges = []

    for i in tqdm(range(num_bpe_merges_max), desc="Training BPE Tokenizer (naive)"):
        pair_freqs = collections.defaultdict(int)

        # --- compute pair frequencys ---
        for word_as_ids, freq in word_freqs.items():
            for i in range(len(word_as_ids) - 1):
                pair = (word_as_ids[i], word_as_ids[i+1])
                pair_freqs[pair] += freq

        if not pair_freqs:
            break

        # --- tie_breaking ---
        best_pair = max(pair_freqs, key=lambda p: (pair_freqs[p], vocab[p[0]], vocab[p[1]]))
        p1, p2 = best_pair

        # --- update word frequencys ---
        word_freqs_updated = collections.defaultdict(int)
        for word_as_ids, freq in word_freqs.items():
            word_as_ids_updated = []
            i = 0
            while i < len(word_as_ids):
                if i < len(word_as_ids) - 1 and word_as_ids[i] == p1 and word_as_ids[i+1] == p2:
                    word_as_ids_updated.append(next_id)
                    i += 2
                else:
                    word_as_ids_updated.append(word_as_ids[i])
                    i += 1
            word_freqs_updated[tuple(word_as_ids_updated)] += freq
        word_freqs = word_freqs_updated

        merges.append((vocab[p1], vocab[p2])) # record merge
        vocab[next_id]  = vocab[p1] + vocab[p2] # append new token to vocabulary
        next_id += 1 # update new token id

    return vocab, merges

def train_bpe_tokenizer_optimized(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    assert vocab_size >= 256 + len(special_tokens)
    
    vocab = {i: bytes([(i)]) for i in range(256)} # vocabulary initialization
    next_id = 256
    
    # --- add special tokens to vocabulary ---
    for s in special_tokens:
        vocab[next_id] = s.encode("utf-8")
        next_id += 1

    # --- pre-tokenization (word frequencys) ---
    word_freqs = collections.defaultdict(int)
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    if special_tokens:
        special_pattern = "|".join(re.escape(s) for s in special_tokens)
        text_chunks = re.split(f'({special_pattern})', text)
    else:
        text_chunks = [text]

    for chunk in text_chunks:
        if chunk in special_tokens:
            continue
        for match in re.finditer(PAT, chunk):
            word_bytes = match.group(0).encode("utf-8")
            word_freqs[tuple(word_bytes)] += 1

    pair_freqs = collections.defaultdict(int)
    pair_to_words_map = collections.defaultdict(set)

    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_freqs[pair] += freq
            pair_to_words_map[pair].add(word)
    
    num_bpe_merges_max = vocab_size - len(vocab)
    merges = []

    for i in tqdm(range(num_bpe_merges_max), desc="Training BPE Tokenizer (optimized)"):
        if not pair_freqs:
            break

        best_pair = max(pair_freqs.keys(), key=lambda p: (pair_freqs[p], vocab[p[0]], vocab[p[1]]))
        p1, p2 = best_pair

        new_id = next_id
        merges.append((vocab[p1], vocab[p2]))
        vocab[new_id] = vocab[p1] + vocab[p2]
        next_id += 1

        words_to_update = list(pair_to_words_map[best_pair])

        for word in words_to_update:
            freq = word_freqs[word]
            del word_freqs[word]

            # 减去旧影响 (Undo)
            for j in range(len(word) - 1):
                pair = (word[j], word[j+1])
                pair_freqs[pair] -= freq
                pair_to_words_map[pair].discard(word) # .discard instead of .remove

                if pair_freqs[pair] == 0:
                    del pair_freqs[pair]
                if not pair_to_words_map[pair]:
                    del pair_to_words_map[pair]

            # 创建新单词 (Modify)
            new_word = []
            k = 0
            while k < len(word):
                if k < len(word) - 1 and word[k] == p1 and word[k+1] == p2:
                    new_word.append(new_id)
                    k += 2
                else:
                    new_word.append(word[k])
                    k += 1
            new_word = tuple(new_word)

            # 增加新影响 (Apply)
            word_freqs[new_word] = freq
            for j in range(len(new_word) - 1):
                pair = (new_word[j], new_word[j+1])
                pair_freqs[pair] += freq
                pair_to_words_map[pair].add(new_word)
        
    return vocab, merges

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / 'tests' / 'fixtures' / 'tinystories_sample_5M.txt'
    vocab_size = 1000
    special_tokens = ["<|endoftext|>"]
    
    start_time_naive = time.time()
    vocab_naive, merges_naive = train_bpe_tokenizer_naive(file_path, vocab_size, special_tokens)
    end_time_naive = time.time()
    duration_naive = end_time_naive - start_time_naive
    print(f"Naive version took: {duration_naive:.2f} seconds")

    start_time_optimized = time.time()
    vocab_optimized, merges_optimized = train_bpe_tokenizer_optimized(file_path, vocab_size, special_tokens)
    end_time_optimized = time.time()
    duration_optimized = end_time_optimized - start_time_optimized
    print(f"Optimized version took: {duration_optimized:.2f} seconds")
    
    if vocab_naive == vocab_optimized:
        print("Vocabularies are identical.")
    if merges_naive == merges_optimized:
        print("Merges are identical.")

    if duration_optimized > 0:
        speedup = duration_naive / duration_optimized
        print(f"Performance Summary: Optimized version is {speedup:.2f}x faster.")