import regex as re
from collections.abc import Iterable, Iterator
import base64
from train_bpe_tokenizer import train_bpe_tokenizer_optimized
from pathlib import Path
import os

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPE_Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.pat = re.compile(PAT)
        self.id_to_byte = vocab
        self.merges = merges

        self.byte_to_id = {v: k for k, v in vocab.items()}

        self.merge_ranks: dict[tuple[int, int], int] = {}
        for i, (p1, p2) in enumerate(self.merges):
            id1 = self.byte_to_id.get(p1)
            id2 = self.byte_to_id.get(p2)
            self.merge_ranks[(id1, id2)] = i

        self.special_tokens: dict[str, int] = {}
        self.special_tokens_pat: re.Pattern | None = None
        
        if special_tokens:  
            for token_str in special_tokens:
                token_bytes = token_str.encode("utf-8")
                if token_bytes not in self.byte_to_id:
                    new_id = len(self.id_to_byte)
                    self.id_to_byte[new_id] = token_bytes
                    self.byte_to_id[token_bytes] = new_id
                self.special_tokens[token_str] = self.byte_to_id[token_bytes]

            escaped_tokens = [re.escape(s) for s in special_tokens]
            self.special_tokens_pat = re.compile(f"({ '|'.join(escaped_tokens) })")

    def _get_pairs(self, ids: list[int]) -> set[tuple[int, int]]:
        pairs = set()
        for i in range(len(ids) - 1):
            pairs.add((ids[i], ids[i+1]))
        return pairs

    def _bpe_merge(self, word_bytes: bytes) -> list[int]:
        if not word_bytes:
            return []
        
        ids = list(word_bytes)

        while len(ids) > 1:
            pairs = self._get_pairs(ids)
            
            best_pair = min(pairs, key=lambda p: self.merge_ranks.get(p, float('inf')))

            if best_pair not in self.merge_ranks:
                break

            p1, p2 = best_pair

            merged_bytes = self.id_to_byte[p1] + self.id_to_byte[p2]
            new_id = self.byte_to_id[merged_bytes]
            
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == p1 and ids[i+1] == p2:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids

        return ids
    
    def encode(self, text: str) -> list[int]:
        token_ids = []

        if self.special_tokens_pat:
            text_chunks = self.special_tokens_pat.split(text)
        else:
            text_chunks = [text]

        for chunk in text_chunks:
            if not chunk:
                continue

            if chunk in self.special_tokens:
                token_ids.append(self.special_tokens[chunk])
            else:
                for match in self.pat.finditer(chunk):
                    word_bytes = match.group(0).encode("utf-8")

                    token_ids.extend(self._bpe_merge(word_bytes))

        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text_chunk in iterable:
            yield from self.encode(text_chunk)

    def decode(self, ids: list[int]) -> str:
        byte_chunks = [self.id_to_byte.get(i) for i in ids]
        full_bytes = b"".join(byte_chunks)

        return full_bytes.decode("utf-8", errors="replace")

    def save(self, vocab_filepath: str, merges_filepath: str):
        with open(vocab_filepath, 'w', encoding='utf-8') as f:
            for token_id, token_bytes in self.id_to_byte.items():
                encoded_bytes = base64.b64encode(token_bytes).decode('ascii')
                f.write(f"{token_id}\t{encoded_bytes}\n")

        with open(merges_filepath, 'w', encoding='utf-8') as f:
            for p1, p2 in self.merges:
                encoded_p1 = base64.b64encode(p1).decode('ascii')
                encoded_p2 = base64.b64encode(p2).decode('ascii')
                f.write(f"{encoded_p1} {encoded_p2}\n")

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> "BPE_Tokenizer":
        vocab = {}
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    token_id_str, encoded_bytes = line.split('\t')
                    token_id = int(token_id_str)
                    token_bytes = base64.b64decode(encoded_bytes)
                    vocab[token_id] = token_bytes

        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    encoded_p1, encoded_p2 = line.split(' ')
                    p1 = base64.b64decode(encoded_p1)
                    p2 = base64.b64decode(encoded_p2)
                    merges.append((p1, p2))

        return cls(vocab, merges, special_tokens)
        
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    file_path_tinystories_train = project_root / 'data' / 'TinyStoriesV2-GPT4-train.txt'
    file_path_tinystories_sample_5m = project_root / 'tests' / 'fixtures' / 'tinystories_sample_5M.txt'

    output_dir = project_root / "bpe_tokenizer_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "vocab.txt"
    merges_path = output_dir / "merges.txt"

    file_path = file_path_tinystories_sample_5m
    vocab_size = 1000
    special_tokens = ["<|endoftext|>"]

    vocab, merges = train_bpe_tokenizer_optimized(file_path, vocab_size, special_tokens)

    bpe_tokenizer = BPE_Tokenizer(vocab, merges, special_tokens)
    bpe_tokenizer.save(vocab_path, merges_path)

    bpe_tokenizer_loaded = BPE_Tokenizer.from_files(vocab_path, merges_path, special_tokens)

    test_text = "hello world, this is a test with a special token <|endoftext|>."

    encoded_ids = bpe_tokenizer_loaded.encode(test_text)
    decoded_text = bpe_tokenizer_loaded.decode(encoded_ids)

    assert test_text == decoded_text
    print("encode/decode succeed!")

    class DummyIterable:
        def __iter__(self):
            yield "hello world,"
            yield " this is a test"
            yield " with a special token <|endoftext|>."

    iterable_encoder = bpe_tokenizer_loaded.encode_iterable(DummyIterable())
    iterable_ids = list(iterable_encoder)

    assert encoded_ids == iterable_ids
    print("iterable encode succeed!")
