import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.train_bpe_tokenizer import train_bpe_tokenizer_naive

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / 'tests' / 'fixtures' / 'tinystories_sample_5M.txt'

    # --- 场景 1: 1个特殊Token, 10次合并 ---
    print("--- SCENARIO 1: 1 Special Token, 10 Merges ---")
    special_tokens_1 = ["<|endoftext|>"]
    num_merges_1 = 10
    vocab_size_1 = 256 + len(special_tokens_1) + num_merges_1
    vocab1, merges1 = train_bpe_tokenizer_naive(file_path, vocab_size_1, special_tokens_1)
    print(f"Final vocab size: {len(vocab1)}\n")

    # --- 场景 2: 2个特殊Token, 9次合并 ---
    print("--- SCENARIO 2: 2 Special Tokens, 9 Merges ---")
    special_tokens_2 = ["<|endoftext|>", "[PAD]"]
    num_merges_2 = 9
    vocab_size_2 = 256 + len(special_tokens_2) + num_merges_2
    vocab2, merges2 = train_bpe_tokenizer_naive(file_path, vocab_size_2, special_tokens_2)
    print(f"Final vocab size: {len(vocab2)}\n")

    # --- 比较结果 ---
    print("="*20 + " COMPARISON " + "="*20)
    
    # 1. 比较合并列表 (Merges)
    print("\n--- Comparing Merge Lists ---")
    print(f"{'#':<3} | {'Scenario 1 (10 merges)':<25} | {'Scenario 2 (9 merges)':<25}")
    print("-" * 60)
    for i in range(len(merges1)):
        merge1_str = f"{merges1[i][0].decode('utf-8', 'ignore')} + {merges1[i][1].decode('utf-8', 'ignore')}"
        if i < len(merges2):
            merge2_str = f"{merges2[i][0].decode('utf-8', 'ignore')} + {merges2[i][1].decode('utf-8', 'ignore')}"
            is_same = "(SAME)" if merges1[i] == merges2[i] else "(DIFFERENT)"
        else:
            merge2_str = "--- (No merge) ---"
            is_same = ""
        print(f"{i+1:<3} | {merge1_str:<25} | {merge2_str:<25} {is_same}")

    # 2. 比较词汇表 (Vocab) 中新增的Token
    print("\n--- Comparing Newly Added Vocab Tokens ---")
    print(f"{'ID':<5} | {'Scenario 1 Vocab':<25} | {'Scenario 2 Vocab':<25}")
    print("-" * 60)
    # 比较特殊Token和合并产生的Token
    for i in range(256, vocab_size_1):
        token1_str = f"{vocab1.get(i, b'N/A')}"
        token2_str = f"{vocab2.get(i, b'N/A')}"
        print(f"{i:<5} | {token1_str:<25} | {token2_str:<25}")