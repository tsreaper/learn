## unicode1

(a) NULL character.

(b) Its string representation is `\\x00`, while its printed representation is `\x00`.

(c) NULL character is not visible when printed.

## unicode2

(a) UTF-8 encoded bytes are shorter and more compacted than UTF-16 and UTF-32.

(b) Wrong example: "你好". Because most characters in UTF-8 are represented by more than 1 byte.

(c) [128, 128]. Because UTF-8 first byte must start with 0, 110, 1110 or 11110.

## train_bpe_tinystories

(a) Training took 53 seconds and reached a peak summed RSS of 1949 MB, measured across the parent process and all pre-tokenization workers. The longest vocabulary token was " accomplishment", which makes sense as it is a valid English word.

(b) The pre-tokenization process, specifically the regexp matches took the longest time.

## train_bpe_expts_owt

(a) Training took 722 seconds and reached a peak summed RSS of 14477 MB, measured across the parent process and all pre-tokenization workers. The longest vocabulary token was "ÃÂÃÂÃÂÃÂ...", which is a mojibake, often seen on websites with encoding problems, so it also makes sense.

(b) TinyStories vocabularies are all English texts, while OpenWebText vocabularies have words from other languages, digits, or even garbled characters.

## tokenizer_experiments

(a) Compression ratio is 4.14 for TinyStories and 4.70 for OpenWebText.

(b) Compression ratio drops to 4.03 for TinyStories and 3.21 for OpenWebText.

(c) Using a single process, the throughput is about 6.69e6 bytes/s. Using all CPU cores, the throughput is about 4.12e7 bytes/s. It will take about 6 hours to tokenize the Pile dataset (825GB of text).

(d) As our vocabulary size is less than $2^{16}$ but larger than $2^8$, `uint16` can hold all the vocabulary without wasting too much memory.
