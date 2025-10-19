import regex as re
import collections
from tqdm import tqdm
from pathlib import Path
import time
from typing import BinaryIO
import os
import multiprocessing

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
            
    return sorted(set(chunk_boundaries))

def _process_chunk(input_path: str, start: int, end: int, pat_str: str, special_tokens: list[str]) -> collections.defaultdict[tuple[int, ...], int]:
    word_freqs = collections.defaultdict(int)

    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)

    text = chunk_bytes.decode("utf-8")

    if special_tokens:
        special_pattern = "|".join(re.escape(s) for s in special_tokens)
        text_chunks = re.split(f'({special_pattern})', text)
    else:
        text_chunks = [text]

    pat = re.compile(pat_str)
    for chunk in text_chunks:
        if chunk in special_tokens:
            continue
        for match in pat.finditer(chunk):
            word_bytes = match.group(0).encode("utf-8")
            word_freqs[tuple(word_bytes)] += 1
    
    return word_freqs

def _serial_pre_tokenization(input_path: str, special_tokens: list[str]) -> collections.defaultdict[tuple[int, ...], int]:
    word_freqs = collections.defaultdict(int)

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    if special_tokens:
        special_pattern = "|".join(re.escape(s) for s in special_tokens)
        text_chunks = re.split(f'({special_pattern})', text)
    else:
        text_chunks = [text]

    pat = re.compile(PAT)
    for chunk in text_chunks:
        if chunk in special_tokens:
            continue
        for match in pat.finditer(chunk):
            word_bytes = match.group(0).encode("utf-8")
            word_freqs[tuple(word_bytes)] += 1
    
    return word_freqs

def _parallel_pre_tokenization(input_path: str, special_tokens: list[str], num_processes: int) -> collections.defaultdict[tuple[int, ...], int]:
    print(f"\n😍 - using {num_processes} processes for parallel pre-tokenization")

    split_token_bytes = special_tokens[0].encode('utf-8')

    with open(input_path, 'rb') as f:
        boundaries = _find_chunk_boundaries(f, num_processes, split_token_bytes)

    num_actual_chunks = len(boundaries) - 1

    tasks = []
    for i in range(num_actual_chunks):
        start_byte = boundaries[i]
        end_byte = boundaries[i+1]
        tasks.append((input_path, start_byte, end_byte, PAT, special_tokens))

    with multiprocessing.Pool(processes=min(num_processes, num_actual_chunks)) as pool:
        list_of_freqs = list(pool.starmap(_process_chunk, tasks))

    total_word_freqs = collections.defaultdict(int)
    for freqs_dict in list_of_freqs:
        for word, freq in freqs_dict.items():
            total_word_freqs[word] += freq

    return total_word_freqs

def train_bpe_tokenizer_naive(input_path: str, vocab_size: int, special_tokens: list[str], num_processes: int = 1) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    assert vocab_size >= 256 + len(special_tokens)
    
    vocab = {i: bytes([(i)]) for i in range(256)}
    next_id = 256
    
    for s in special_tokens:
        vocab[next_id] = s.encode("utf-8")
        next_id += 1

    if num_processes > 1:
        word_freqs = _parallel_pre_tokenization(input_path, special_tokens, num_processes)
    else:
        word_freqs = _serial_pre_tokenization(input_path, special_tokens)
                
    num_bpe_merges_max = vocab_size - len(vocab)
    merges = []

    for i in tqdm(range(num_bpe_merges_max), desc="training bpe tokenizer (naive)"):
        pair_freqs = collections.defaultdict(int)

        for word_as_ids, freq in word_freqs.items():
            for i in range(len(word_as_ids) - 1):
                pair = (word_as_ids[i], word_as_ids[i+1])
                pair_freqs[pair] += freq

        if not pair_freqs:
            break

        best_pair = max(pair_freqs, key=lambda p: (pair_freqs[p], vocab[p[0]], vocab[p[1]]))
        p1, p2 = best_pair

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

        merges.append((vocab[p1], vocab[p2]))
        vocab[next_id]  = vocab[p1] + vocab[p2]
        next_id += 1

    return vocab, merges

def train_bpe_tokenizer_optimized(input_path: str, vocab_size: int, special_tokens: list[str], num_processes: int = 1) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    assert vocab_size >= 256 + len(special_tokens)
    
    vocab = {i: bytes([(i)]) for i in range(256)}
    next_id = 256
    
    for s in special_tokens:
        vocab[next_id] = s.encode("utf-8")
        next_id += 1

    if num_processes > 1:
        word_freqs = _parallel_pre_tokenization(input_path, special_tokens, num_processes)
    else:
        word_freqs = _serial_pre_tokenization(input_path, special_tokens)

    pair_freqs = collections.defaultdict(int)
    pair_to_words_map = collections.defaultdict(set)

    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_freqs[pair] += freq
            pair_to_words_map[pair].add(word)
    
    num_bpe_merges_max = vocab_size - len(vocab)
    merges = []

    for i in tqdm(range(num_bpe_merges_max), desc="training bpe tokenizer (optimized)"):
        if not pair_freqs:
            break

        best_pair = max(pair_freqs.keys(), key=lambda p: (pair_freqs[p], vocab[p[0]], vocab[p[1]]))
        p1, p2 = best_pair

        merges.append((vocab[p1], vocab[p2]))
        new_id = next_id
        vocab[new_id] = vocab[p1] + vocab[p2]
        next_id += 1

        words_to_update = list(pair_to_words_map[best_pair])

        for word in words_to_update:
            freq = word_freqs[word]
            del word_freqs[word]

            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                pair_freqs[pair] -= freq
                pair_to_words_map[pair].discard(word) # .discard() instead of .remove()

                if pair_freqs[pair] == 0:
                    del pair_freqs[pair]
                if not pair_to_words_map[pair]:
                    del pair_to_words_map[pair]

            new_word = []
            i = 0

            while i < len(word):
                if i < len(word) - 1 and word[i] == p1 and word[i+1] == p2:
                    new_word.append(new_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_word = tuple(new_word)

            word_freqs[new_word] = freq

            for i in range(len(new_word) - 1):
                pair = (new_word[i], new_word[i+1])
                pair_freqs[pair] += freq
                pair_to_words_map[pair].add(new_word)
        
    return vocab, merges

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    file_path_tinystories_train = project_root / 'data' / 'TinyStoriesV2-GPT4-train.txt'
    file_path_tinystories_sample_5m = project_root / 'tests' / 'fixtures' / 'tinystories_sample_5M.txt'

    file_path = file_path_tinystories_sample_5m
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    parallel_pre_tokenization = True

    if parallel_pre_tokenization:
        num_processes = os.cpu_count() // 2
    else:
        num_processes = 1
    
    start_time_naive = time.time()
    vocab_naive, merges_naive = train_bpe_tokenizer_naive(file_path, vocab_size, special_tokens, num_processes)
    end_time_naive = time.time()
    duration_naive = end_time_naive - start_time_naive
    print(f"Naive version took: {duration_naive:.2f} seconds\n")

    start_time_optimized = time.time()
    vocab_optimized, merges_optimized = train_bpe_tokenizer_optimized(file_path, vocab_size, special_tokens, num_processes)
    end_time_optimized = time.time()
    duration_optimized = end_time_optimized - start_time_optimized
    print(f"Optimized version took: {duration_optimized:.2f} seconds\n")
    
    if vocab_naive == vocab_optimized:
        print("Vocabularies are identical.")
    if merges_naive == merges_optimized:
        print("Merges are identical.\n")

    if duration_optimized > 0:
        speedup = duration_naive / duration_optimized
        print(f"Performance Summary: Optimized version is {speedup:.2f}x faster.")