"""Experimental deterministic byte-level BPE tokenizer.

This tokenizer is intentionally separate from the production ByteTokenizer and
is not wired into model training or checkpoint loading yet.
"""
from __future__ import annotations

import hashlib
import heapq
import hmac
import json
import os
import re
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path


class ByteLevelBPETokenizer:
    """Trainable UTF-8 byte BPE with a fixed 256-token byte alphabet."""

    name = "utf8-byte-bpe-v1"
    base_vocab_size = 256
    format_version = 1

    def __init__(self, merges: Iterable[Sequence[int]] = ()):
        self._merges: list[tuple[int, int]] = []
        self._merge_ranks: dict[tuple[int, int], int] = {}
        self._token_bytes: list[bytes] = [bytes((value,)) for value in range(256)]

        for rank, pair in enumerate(merges):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("Each merge must contain exactly two token IDs.")
            left, right = pair
            if type(left) is not int or type(right) is not int:
                raise TypeError("Merge token IDs must be integers; booleans are invalid.")
            new_token_id = self.base_vocab_size + rank
            if not (0 <= left < new_token_id and 0 <= right < new_token_id):
                raise ValueError("A merge may only reference tokens that already exist.")
            if (left, right) in self._merge_ranks:
                raise ValueError("Duplicate merge pair.")
            self._merge_ranks[(left, right)] = rank
            self._merges.append((left, right))
            self._token_bytes.append(self._token_bytes[left] + self._token_bytes[right])

    @property
    def vocab_size(self) -> int:
        return self.base_vocab_size + len(self._merges)

    @property
    def merges(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._merges)

    @classmethod
    def train(
        cls,
        texts: str | Iterable[str],
        *,
        max_vocab_size: int = 2048,
        min_frequency: int = 2,
    ) -> "ByteLevelBPETokenizer":
        """Learn merges deterministically without merging across document boundaries."""
        if type(max_vocab_size) is not int or max_vocab_size < cls.base_vocab_size:
            raise ValueError("max_vocab_size must be an integer >= 256.")
        if type(min_frequency) is not int or min_frequency < 1:
            raise ValueError("min_frequency must be a positive integer.")

        if isinstance(texts, str):
            documents = [texts]
        else:
            documents = list(texts)
        if any(not isinstance(text, str) for text in documents):
            raise TypeError("Every training document must be str.")

        tokens: list[int] = []
        previous: list[int] = []
        following: list[int] = []
        active: list[bool] = []
        for document in documents:
            encoded = document.encode("utf-8")
            start = len(tokens)
            tokens.extend(encoded)
            count = len(encoded)
            for offset in range(count):
                node = start + offset
                previous.append(node - 1 if offset else -1)
                following.append(node + 1 if offset + 1 < count else -1)
                active.append(True)

        pair_counts: dict[tuple[int, int], int] = {}
        pair_positions: dict[tuple[int, int], set[int]] = {}
        pair_versions: dict[tuple[int, int], int] = {}
        heap: list[tuple[int, int, int, int]] = []

        def adjust_edge(left_node: int, right_node: int, delta: int) -> None:
            if left_node < 0 or right_node < 0:
                return
            pair = (tokens[left_node], tokens[right_node])
            old_count = pair_counts.get(pair, 0)
            new_count = old_count + delta
            if new_count < 0:
                raise RuntimeError("Internal BPE pair-count invariant was violated.")

            positions = pair_positions.setdefault(pair, set())
            if delta > 0:
                positions.add(left_node)
            else:
                positions.discard(left_node)

            version = pair_versions.get(pair, 0) + 1
            pair_versions[pair] = version
            if new_count:
                pair_counts[pair] = new_count
                heapq.heappush(heap, (-new_count, pair[0], pair[1], version))
            else:
                pair_counts.pop(pair, None)
                pair_positions.pop(pair, None)

        for node, right_node in enumerate(following):
            if right_node >= 0:
                adjust_edge(node, right_node, 1)

        learned_merges: list[tuple[int, int]] = []
        while cls.base_vocab_size + len(learned_merges) < max_vocab_size:
            selected_pair = None
            while heap:
                negative_count, left_id, right_id, version = heapq.heappop(heap)
                pair = (left_id, right_id)
                if (
                    pair_versions.get(pair) == version
                    and pair_counts.get(pair) == -negative_count
                ):
                    selected_pair = pair
                    selected_count = -negative_count
                    break
            if selected_pair is None or selected_count < min_frequency:
                break

            # Snapshot the old occurrences. Newly-created occurrences belong to
            # the next merge iteration, as required by standard BPE merge rounds.
            candidate_nodes = sorted(pair_positions[selected_pair])
            new_token_id = cls.base_vocab_size + len(learned_merges)
            merged_occurrences = 0

            for left_node in candidate_nodes:
                right_node = following[left_node]
                if (
                    not active[left_node]
                    or right_node < 0
                    or not active[right_node]
                    or (tokens[left_node], tokens[right_node]) != selected_pair
                ):
                    continue

                before_node = previous[left_node]
                after_node = following[right_node]

                adjust_edge(before_node, left_node, -1)
                adjust_edge(left_node, right_node, -1)
                adjust_edge(right_node, after_node, -1)

                tokens[left_node] = new_token_id
                following[left_node] = after_node
                if after_node >= 0:
                    previous[after_node] = left_node
                active[right_node] = False
                previous[right_node] = -1
                following[right_node] = -1

                adjust_edge(before_node, left_node, 1)
                adjust_edge(left_node, after_node, 1)
                merged_occurrences += 1

            if merged_occurrences == 0:
                raise RuntimeError("Selected BPE pair had no active occurrences.")
            learned_merges.append(selected_pair)

        return cls(learned_merges)

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        raw = text.encode("utf-8")
        if not raw:
            return []

        tokens = list(raw)
        count = len(tokens)
        previous = [index - 1 for index in range(count)]
        following = [index + 1 for index in range(count)]
        following[-1] = -1
        active = [True] * count
        edge_versions = [0] * count
        heap: list[tuple[int, int, int, int]] = []

        def push_edge(left_node: int) -> None:
            right_node = following[left_node]
            if right_node < 0:
                return
            rank = self._merge_ranks.get((tokens[left_node], tokens[right_node]))
            if rank is not None:
                heapq.heappush(
                    heap, (rank, left_node, right_node, edge_versions[left_node])
                )

        for node in range(count - 1):
            push_edge(node)

        while heap:
            rank, left_node, right_node, version = heapq.heappop(heap)
            if (
                not active[left_node]
                or not active[right_node]
                or edge_versions[left_node] != version
                or following[left_node] != right_node
                or self._merge_ranks.get((tokens[left_node], tokens[right_node])) != rank
            ):
                continue

            before_node = previous[left_node]
            after_node = following[right_node]
            if before_node >= 0:
                # The left token changes, invalidating the preceding edge.
                edge_versions[before_node] += 1

            tokens[left_node] = self.base_vocab_size + rank
            following[left_node] = after_node
            edge_versions[left_node] += 1
            if after_node >= 0:
                previous[after_node] = left_node
            active[right_node] = False
            previous[right_node] = -1
            following[right_node] = -1
            edge_versions[right_node] += 1

            if before_node >= 0:
                push_edge(before_node)
            push_edge(left_node)

        result = []
        node = 0
        while node >= 0:
            if active[node]:
                result.append(tokens[node])
            node = following[node]
        return result

    def decode(self, token_ids: Sequence[int]) -> str:
        if not isinstance(token_ids, Sequence) or isinstance(token_ids, str):
            raise TypeError("token_ids must be a sequence of integers")
        pieces = []
        for token_id in token_ids:
            if type(token_id) is not int:
                raise TypeError("Token IDs must be integers; booleans are invalid.")
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(f"Token ID {token_id} is outside this tokenizer vocabulary.")
            pieces.append(self._token_bytes[token_id])
        return b"".join(pieces).decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        payload = {
            "format_version": self.format_version,
            "tokenizer": self.name,
            "base_vocab_size": self.base_vocab_size,
            "vocab_size": self.vocab_size,
            "merges": [list(pair) for pair in self._merges],
        }
        canonical = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        payload["sha256"] = hashlib.sha256(canonical).hexdigest()
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "ByteLevelBPETokenizer":
        required_keys = {
            "format_version", "tokenizer", "base_vocab_size",
            "vocab_size", "merges", "sha256",
        }
        if not isinstance(payload, dict) or set(payload) != required_keys:
            raise ValueError("Tokenizer artifact has an invalid schema.")
        if (
            type(payload["format_version"]) is not int
            or payload["format_version"] != cls.format_version
            or payload["tokenizer"] != cls.name
            or type(payload["base_vocab_size"]) is not int
            or payload["base_vocab_size"] != cls.base_vocab_size
            or type(payload["vocab_size"]) is not int
            or not isinstance(payload["merges"], list)
            or not isinstance(payload["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", payload["sha256"]) is None
        ):
            raise ValueError("Tokenizer artifact metadata is invalid.")

        tokenizer = cls(payload["merges"])
        if tokenizer.vocab_size != payload["vocab_size"]:
            raise ValueError("Tokenizer artifact vocabulary size does not match its merges.")
        expected_hash = tokenizer.to_dict()["sha256"]
        if not hmac.compare_digest(expected_hash, payload["sha256"]):
            raise ValueError("Tokenizer artifact SHA-256 integrity check failed.")
        return tokenizer

    def save(self, path: Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n",
                prefix=f".{destination.name}.", suffix=".tmp",
                dir=destination.parent, delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @classmethod
    def load(cls, path: Path) -> "ByteLevelBPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)
