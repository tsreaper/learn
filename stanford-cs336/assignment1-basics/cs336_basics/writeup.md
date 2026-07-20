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
