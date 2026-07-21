import heapq
import os
import regex as re
import sys
import time
import json
from typing import Self
from collections import defaultdict
from math import ceil
from concurrent.futures import ProcessPoolExecutor, as_completed

from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
ENCODING = "utf-8"


def pretokenization(
    input_path: str,
    special_tokens: list[str],
    start: int,
    end: int
) -> dict[bytes, int]:
    ret: dict[bytes, int] = defaultdict(int)

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        chunk = chunk_bytes.decode(ENCODING, errors="ignore")
        splits = re.split("|".join(map(re.escape, special_tokens)), chunk)

        for split in splits:
            for m in re.finditer(PAT, split):
                ret[m.group().encode(ENCODING)] += 1

    return ret


class MaxHeapItem:
    def __init__(self, val: tuple[int, tuple[bytes, bytes]]):
        self.val = val

    def __lt__(self, other: Self):
        return self.val > other.val


class MaxHeap:
    def __init__(self):
        self.heap: list[MaxHeapItem] = []
        self.deleted: list[MaxHeapItem] = []

    def push(self, item: tuple[int, tuple[bytes, bytes]]):
        heapq.heappush(self.heap, MaxHeapItem(item))

    def remove(self, item: tuple[int, tuple[bytes, bytes]]):
        heapq.heappush(self.deleted, MaxHeapItem(item))

    def top(self) -> tuple[int, tuple[bytes, bytes]]:
        while True:
            if len(self.deleted) == 0 or self.heap[0].val != self.deleted[0].val:
                break
            heapq.heappop(self.heap)
            heapq.heappop(self.deleted)
        return self.heap[0].val


class Index:
    def __init__(self, seqs: list[list[int]], cnts: list[int]):
        self.total_cnt: dict[tuple[int, int], int] = defaultdict(int)
        self.detail: dict[
            tuple[int, int], dict[int, int]
        ] = defaultdict(lambda: defaultdict(int))

        for i in range(len(seqs)):
            for j in range(1, len(seqs[i])):
                p = (seqs[i][j - 1], seqs[i][j])
                self.total_cnt[p] += cnts[i]
                self.detail[p][i] += cnts[i]

    def update(self, pair: tuple[int, int], pos: int, det: int):
        self.total_cnt[pair] += det
        self.detail[pair][pos] += det
        if self.detail[pair][pos] == 0:
            del self.detail[pair][pos]


def bpe(
    vocab_size: int,
    special_tokens: list[str],
    bytes_count: dict[bytes, int]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    seqs = list(map(list, bytes_count.keys()))
    cnts = list(bytes_count.values())
    index = Index(seqs, cnts)

    vocab: dict[int, bytes] = {}
    vocab_reversed: dict[bytes, int] = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for special_token in special_tokens:
        idx = len(vocab)
        vocab[idx] = special_token.encode(ENCODING)
    for idx, token in vocab.items():
        vocab_reversed[token] = idx

    heap = MaxHeap()
    for pair, cnt in index.total_cnt.items():
        token_pair = (vocab[pair[0]], vocab[pair[1]])
        heap.push((cnt, token_pair))

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size:
        cnt, (tx, ty) = heap.top()
        x, y = vocab_reversed[tx], vocab_reversed[ty]
        merges.append((tx, ty))
        new_idx = len(vocab)
        vocab[new_idx] = tx + ty
        vocab_reversed[tx + ty] = new_idx

        total_changes: dict[tuple[int, int], int] = defaultdict(int)
        positions = list(index.detail[(x, y)].keys())
        for i in positions:
            changes: dict[tuple[int, int], int] = defaultdict(int)
            new_seq = [seqs[i][0]]
            for j in range(1, len(seqs[i])):
                if new_seq[-1] == x and seqs[i][j] == y:
                    changes[(x, y)] -= cnts[i]
                    if len(new_seq) > 1:
                        z = new_seq[-2]
                        changes[(z, x)] -= cnts[i]
                        changes[(z, new_idx)] += cnts[i]
                    if j + 1 < len(seqs[i]):
                        z = seqs[i][j + 1]
                        changes[(y, z)] -= cnts[i]
                        changes[(new_idx, z)] += cnts[i]
                    new_seq[-1] = new_idx
                else:
                    new_seq.append(seqs[i][j])
            for pair, det in changes.items():
                index.update(pair, i, det)
                total_changes[pair] += det
            seqs[i] = new_seq

        for pair, det in total_changes.items():
            new_cnt = index.total_cnt[pair]
            old_cnt = new_cnt - det
            token_pair = (vocab[pair[0]], vocab[pair[1]])
            if old_cnt > 0:
                heap.remove((old_cnt, token_pair))
            if new_cnt > 0:
                heap.push((new_cnt, token_pair))

    return vocab, merges


def train(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        n_chunks = int(ceil(file_size / (16 * 1024 * 1024)))
        boundaries = find_chunk_boundaries(
            f, n_chunks, special_tokens[0].encode(ENCODING))

    bytes_count: dict[bytes, int] = defaultdict(int)
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                pretokenization, input_path, special_tokens, start, end)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        for future in as_completed(futures):
            result = future.result()
            for token, cnt in result.items():
                bytes_count[token] += cnt

    return bpe(vocab_size, special_tokens, bytes_count)


class Serializer:
    def __init__(self):
        pass

    def path(self, prefix: str) -> str:
        return "data/%s_bpe.json" % prefix

    def serialize(
        self,
        prefix: str,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]]
    ):
        serialized_vocab = {}
        for idx, token in vocab.items():
            serialized_vocab[idx] = {
                "token": token.decode(ENCODING, errors="replace"),
                "hex": token.hex()
            }
        serialized_merges: list[list[str]] = []
        for pair in merges:
            serialized_merges.append([pair[0].hex(), pair[1].hex()])

        with open(self.path(prefix), "w", encoding=ENCODING) as f:
            json.dump({
                "vocab": serialized_vocab,
                "merges": serialized_merges
            }, f, indent=2)

    def deserialize(self, prefix: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        with open(self.path(prefix), "r", encoding=ENCODING) as f:
            serialized = json.load(f)

            serialized_vocab: dict[str, dict[str, str]] = serialized["vocab"]
            vocab: dict[int, bytes] = {}
            for k, v in serialized_vocab.items():
                vocab[int(k)] = bytes.fromhex(v["hex"])

            serialized_merges: list[list[str]] = serialized["merges"]
            merges: list[tuple[bytes, bytes]] = []
            for pair in serialized_merges:
                merges.append((bytes.fromhex(pair[0]), bytes.fromhex(pair[1])))

            return vocab, merges


if __name__ == "__main__":
    vocab, merges, arg = {}, [], ""
    if len(sys.argv) == 2:
        arg = sys.argv[1]
        start_time = time.time()
        if arg == "tinystories_train":
            vocab, merges = train(
                "data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
        elif arg == "tinystories_valid":
            vocab, merges = train(
                "data/TinyStoriesV2-GPT4-valid.txt", 10000, ["<|endoftext|>"])
        elif arg == "owt_train":
            vocab, merges = train(
                "data/owt_train.txt", 32000, ["<|endoftext|>"])
        elif arg == "owt_valid":
            vocab, merges = train(
                "data/owt_valid.txt", 32000, ["<|endoftext|>"])
        print(
            "BPE training finished in %.3f seconds" % (time.time() - start_time))
    if len(vocab) == 0:
        print("Usage: uv run python cs336_basics/train_bpe.py <dataset>")
        print("<dataset> should be \"tinystories_train\", \"tinystories_valid\", \"owt_train\" or \"owt_valid\"")
    else:
        serializer = Serializer()
        serializer.serialize(arg, vocab, merges)
