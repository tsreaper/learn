import heapq
import regex as re
import os
import shutil
import time
import sys
import numpy as np
from math import ceil
from typing import Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor, as_completed

from cs336_basics.train_bpe import PAT, ENCODING, Serializer
from cs336_basics.pretokenization_example import find_chunk_boundaries


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab

        self.vocab_reversed: dict[bytes, int] = {}
        for idx, token in vocab.items():
            self.vocab_reversed[token] = idx
        self.merges: dict[tuple[int, int], tuple[int, int]] = {}
        for i in range(len(merges)):
            x = self.vocab_reversed[merges[i][0]]
            y = self.vocab_reversed[merges[i][1]]
            z = self.vocab_reversed[merges[i][0] + merges[i][1]]
            self.merges[(x, y)] = (i, z)

        self.special_tokens: set[str] | None = None
        if special_tokens is not None:
            self.special_tokens = set(special_tokens)
            for token in self.special_tokens:
                b = token.encode(ENCODING)
                if b not in self.vocab_reversed.keys():
                    idx = len(self.vocab)
                    self.vocab[idx] = b
                    self.vocab_reversed[b] = idx

    def encode(self, text: str) -> list[int]:
        result: list[int] = []
        splits = [text]
        if self.special_tokens is not None:
            splits = re.split(
                "(" + "|".join(map(re.escape, sorted(self.special_tokens, reverse=True))) + ")", text)

        encodings: dict[str, list[int]] = {}
        for split in splits:
            if self.special_tokens is not None and split in self.special_tokens:
                result.append(self.vocab_reversed[split.encode(ENCODING)])
            else:
                for m in re.finditer(PAT, split):
                    pretoken = m.group(0)
                    if pretoken in encodings.keys():
                        result += encodings[pretoken]
                    else:
                        if len(pretoken) <= 100:
                            encoding = self.encode_pretoken_brute(pretoken)
                        else:
                            encoding = self.encode_pretoken_heap(pretoken)
                        encodings[pretoken] = encoding
                        result += encoding
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for x in self.encode(text):
                yield x

    def encode_pretoken_brute(self, pretoken: str) -> list[int]:
        seq: list[int] = []
        for b in pretoken.encode(ENCODING):
            seq.append(self.vocab_reversed[bytes([b])])

        while True:
            pos, mn, new_idx = -1, len(self.merges), -1
            for i in range(len(seq) - 1):
                key = (seq[i], seq[i + 1])
                if key in self.merges.keys():
                    priority, ni = self.merges[key]
                    if priority < mn:
                        pos, mn, new_idx = i, priority, ni
            if pos < 0:
                break
            new_seq = seq[:pos]
            new_seq.append(new_idx)
            new_seq += seq[pos + 2:]
            seq = new_seq

        return seq

    def encode_pretoken_heap(self, pretoken: str) -> list[int]:
        seq: list[int] = []
        for b in pretoken.encode(ENCODING):
            seq.append(self.vocab_reversed[bytes([b])])
        prv = [i - 1 for i in range(len(seq))]
        nxt = [i + 1 for i in range(len(seq))]
        heap: list[tuple[int, int, int, int, int]] = []

        def add(l: int, r: int):
            key = (seq[l], seq[r])
            if key in self.merges.keys():
                priority = self.merges[key][0]
                heapq.heappush(heap, (priority, l, r, key[0], key[1]))

        for i in range(len(seq) - 1):
            add(i, i + 1)

        while len(heap) > 0:
            _, l, r, vl, vr = heapq.heappop(heap)
            if seq[l] == vl and seq[r] == vr:
                seq[l] = -1
                seq[r] = self.merges[(vl, vr)][1]
                prv[r] = prv[l]

                if prv[r] >= 0:
                    nxt[prv[r]] = r
                    add(prv[r], r)
                if nxt[r] < len(seq):
                    add(r, nxt[r])

        result: list[int] = []
        for x in seq:
            if x >= 0:
                result.append(x)
        return result

    def decode(self, ids: list[int]) -> str:
        result: bytes = bytes()
        for idx in ids:
            result += self.vocab[idx]
        return result.decode(ENCODING, errors="replace")


def tokenize_part(
    input_path: str,
    start: int,
    end: int,
    tokenizer: Tokenizer,
    result_path: str
):
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        chunk = chunk_bytes.decode(ENCODING, errors="ignore")
        result = tokenizer.encode(chunk)
        np_array = np.array(result, dtype=np.uint16)
        np.save(result_path, np_array)


def tokenize(input_path: str, prefix: str):
    dir_path = "data/%s_encoding" % prefix
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

    special_token = "<|endoftext|>"
    with open(input_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        n_chunks = int(ceil(file_size / (16 * 1024 * 1024)))
        boundaries = find_chunk_boundaries(
            f, n_chunks, special_token.encode(ENCODING))

    serializer = Serializer()
    vocab, merges = serializer.deserialize(prefix)
    tokenizer = Tokenizer(vocab, merges, [special_token])
    with ProcessPoolExecutor() as executor:
        futures: list[Future[None]] = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            futures.append(
                executor.submit(
                    tokenize_part, input_path, start, end, tokenizer, "%s/%d.npy" % (dir_path, i)))
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    input_path, arg = "", ""

    if len(sys.argv) == 2:
        arg = sys.argv[1]
        if arg == "tinystories_train":
            input_path = "data/TinyStoriesV2-GPT4-train.txt"
        elif arg == "tinystories_valid":
            input_path = "data/TinyStoriesV2-GPT4-valid.txt"
        elif arg == "owt_train":
            input_path = "data/owt_train.txt"
        elif arg == "owt_valid":
            input_path = "data/owt_valid.txt"
    if len(arg) == 0:
        print("Usage: uv run python cs336_basics/tokenizer.py <dataset>")
        print("<dataset> should be \"tinystories_train\", \"tinystories_valid\", \"owt_train\" or \"owt_valid\"")
    else:
        start_time = time.time()
        tokenize(input_path, arg)
        print("Tokenization finished in %.3f seconds" %
              (time.time() - start_time))
