import json

import regex as re
from typing import Iterable, Iterator, Any

class Tokenizer:
    BYTE_TOKENS = tuple(bytes([i]) for i in range(256))

    def __init__(
        self, 
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = special_tokens or []
        next_id = max(self.vocab.keys()) + 1 if self.vocab else 0
        self.bytes_to_id: dict[bytes, int] = {b:i for i, b in self.vocab.items()}

        self.special_token_to_id: dict[str, int] = {}
        for special_token in self.special_tokens:
            special_token_bytes = special_token.encode("utf-8")
            if special_token_bytes in self.bytes_to_id:
                # pass
                special_token_id = self.bytes_to_id[special_token_bytes]
            else:
                self.vocab[next_id] = special_token_bytes
                self.bytes_to_id[special_token_bytes] = next_id
                special_token_id = next_id
                next_id += 1
            self.special_token_to_id[special_token] = special_token_id

        if self.special_tokens:
            escaped = sorted((re.escape(tok) for tok in self.special_tokens), key=len, reverse=True)
            self.pattern_pre = "|".join(escaped)
        else:
            self.pattern_pre = None
        
    @classmethod
    def from_files(cls, vocab_filepath, merge_filepath, special_tokens=None):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_payload = json.load(f)

        vocab: dict[int, bytes] = {}
        for item in vocab_payload:
            token_id = int(item["id"])
            token_bytes = bytes.fromhex(item["bytes_hex"])
            vocab[token_id] = token_bytes

        with open(merge_filepath, "r", encoding="utf-8") as f:
            merges_payload = json.load(f)

        merges: list[tuple[bytes, bytes]] = []
        for item in merges_payload:
            left = bytes.fromhex(item["left_hex"])
            right = bytes.fromhex(item["right_hex"])
            merges.append((left, right))

        return cls(vocab, merges, special_tokens)

    def encode(
        self, 
        text: str
    ) -> list[int]:
        # handle special token 
        if self.pattern_pre is None:
            pieces = [text]
        else:
            pieces = re.split(f"({self.pattern_pre})", text)

        ids: list[int] = []
        # every piece is a long text with space, we need to use pattern to spilt
        for piece in pieces:
            if piece == "": 
                continue
            if piece in self.special_token_to_id:
                ids.append(self.special_token_to_id[piece])
            else:
                pretokenized_word = self._pretokenize(piece)
                for pretoken in pretokenized_word:
                    ids.extend(self._encode_pretoken(pretoken))
        return ids
    
    def _apply_bpe(self, pieces: list[bytes]) -> list[bytes]:
        # pieces is a pretokend word, merge it
        pieces = list(pieces)
        while len(pieces) > 1:
            merged = False
            for left_tok, right_tok in self.merges:
                for i in range(len(pieces) - 1):
                    if pieces[i] == left_tok and pieces[i + 1] == right_tok:
                        merged_piece = pieces[i] + pieces[i + 1]
                        pieces = pieces[:i] + [merged_piece] + pieces[i + 2:]
                        merged = True
                        break
                if merged:
                    break
            if not merged:
                break
        return pieces

    def _encode_pretoken(self, pretoken_bytes: list[bytes]) -> list[int]:
        merged_pieces = self._apply_bpe(pretoken_bytes)
        ans: list[int] = []
        for m in merged_pieces:
            if m not in self.bytes_to_id:
                raise ValueError(f"Token bytes not in vocab: {m!r}")
            ans.append(self.bytes_to_id[m])
        return ans
        
    def _pretokenize(
        self, 
        text: str
    ) -> list[list[bytes]]:
        pretoken_text_bytes = []
        word = re.finditer(self.pattern, text)
        for w in word: 
            pretoken = w.group(0)
            pretoken_bytes = list(Tokenizer.BYTE_TOKENS[b] for b in pretoken.encode("utf-8"))
            pretoken_text_bytes.append(pretoken_bytes)
        return pretoken_text_bytes
        
    def encode_iterable(
        self, 
        iterable: Iterable[str]
    ) -> Iterator[int]:
        for chunk in iterable: 
            yield from self.encode(chunk)
    
    def decode(self, ids: list[int]) -> str:
        string_bytes = bytearray()
        replacement = "\uFFFD".encode("utf-8")

        for token_id in ids:
            if token_id not in self.vocab:
                string_bytes.extend(replacement)
            else:
                string_bytes.extend(self.vocab[token_id])

        return bytes(string_bytes).decode("utf-8", errors="replace")


def my_get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab, merges, special_tokens)