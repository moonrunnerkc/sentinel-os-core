# Author: Bradley R. Kinnard
# merkle tree for efficient batch verification and commitment

import hashlib
from dataclasses import dataclass
from typing import Any

from utils.helpers import get_logger

logger = get_logger(__name__)


@dataclass
class MerkleProof:
    """proof of inclusion in a merkle tree."""
    leaf: str
    leaf_index: int
    siblings: list[tuple[str, str]]  # (hash, position: "left" or "right")
    root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "leaf": self.leaf,
            "leaf_index": self.leaf_index,
            "siblings": self.siblings,
            "root": self.root,
        }


class MerkleTree:
    """
    merkle tree for efficient batch commitment and verification.

    used for:
    - batch ZK proofs
    - log integrity verification
    - state checkpoint commitments
    """

    def __init__(self):
        self._leaves: list[str] = []
        self._tree: list[list[str]] = []
        self._built = False

    @staticmethod
    def _hash(left: str, right: str) -> str:
        """hash two nodes together."""
        combined = left + right
        return hashlib.sha256(combined.encode()).hexdigest()

    @staticmethod
    def _hash_leaf(data: str) -> str:
        """hash a leaf node (prefixed to prevent second preimage attacks)."""
        prefixed = "leaf:" + data
        return hashlib.sha256(prefixed.encode()).hexdigest()

    def add_leaf(self, data: str) -> int:
        """add a leaf and return its index."""
        if self._built:
            raise RuntimeError("cannot add leaves after tree is built")

        self._leaves.append(self._hash_leaf(data))
        return len(self._leaves) - 1

    def add_leaves(self, data: list[str]) -> list[int]:
        """add multiple leaves."""
        return [self.add_leaf(d) for d in data]

    def build(self) -> str:
        """build the tree and return the root."""
        if not self._leaves:
            self._tree = [[]]
            self._built = True
            return ""

        # pad to power of 2
        leaves = self._leaves.copy()
        while len(leaves) & (len(leaves) - 1):
            leaves.append(leaves[-1])  # duplicate last leaf

        self._tree = [leaves]

        # build levels bottom-up
        current = leaves
        while len(current) > 1:
            next_level = []
            for i in range(0, len(current), 2):
                left = current[i]
                right = current[i + 1] if i + 1 < len(current) else current[i]
                next_level.append(self._hash(left, right))
            self._tree.append(next_level)
            current = next_level

        self._built = True
        logger.debug(f"built merkle tree with {len(self._leaves)} leaves, root={self.root[:16]}...")
        return self.root

    @property
    def root(self) -> str:
        """return the merkle root."""
        if not self._built:
            raise RuntimeError("tree not built")
        if not self._tree or not self._tree[-1]:
            return ""
        return self._tree[-1][0]

    def get_proof(self, leaf_index: int) -> MerkleProof:
        """generate inclusion proof for a leaf."""
        if not self._built:
            raise RuntimeError("tree not built")
        if leaf_index < 0 or leaf_index >= len(self._leaves):
            raise IndexError(f"leaf index {leaf_index} out of range")

        siblings = []
        idx = leaf_index

        # pad index for power-of-2 tree
        tree_leaves = self._tree[0]

        for level in range(len(self._tree) - 1):
            layer = self._tree[level]

            if idx % 2 == 0:
                # sibling is on the right
                sibling_idx = idx + 1 if idx + 1 < len(layer) else idx
                siblings.append((layer[sibling_idx], "right"))
            else:
                # sibling is on the left
                siblings.append((layer[idx - 1], "left"))

            idx = idx // 2

        return MerkleProof(
            leaf=self._leaves[leaf_index],
            leaf_index=leaf_index,
            siblings=siblings,
            root=self.root,
        )

    @staticmethod
    def verify_proof(proof: MerkleProof) -> bool:
        """verify a merkle inclusion proof."""
        current = proof.leaf

        for sibling_hash, position in proof.siblings:
            if position == "left":
                current = MerkleTree._hash(sibling_hash, current)
            else:
                current = MerkleTree._hash(current, sibling_hash)

        return current == proof.root

    def verify_all_leaves(self) -> bool:
        """verify all leaves are included (self-consistency check)."""
        if not self._built:
            return False

        for i in range(len(self._leaves)):
            proof = self.get_proof(i)
            if not self.verify_proof(proof):
                return False

        return True


class IncrementalMerkleTree:
    """
    incrementally buildable merkle tree for streaming data.

    allows adding leaves and updating root without full rebuild.
    useful for real-time log commitment.
    """

    def __init__(self, depth: int = 32):
        self._depth = depth
        self._zero_hashes = self._compute_zero_hashes()
        self._leaves: list[str] = []
        self._filled_subtrees: list[str] = ["" for _ in range(depth)]
        self._next_index = 0
        self._root = self._zero_hashes[-1]

    def _compute_zero_hashes(self) -> list[str]:
        """compute zero hashes for empty subtrees."""
        zeros = ["0" * 64]  # zero leaf
        for _ in range(self._depth):
            zeros.append(MerkleTree._hash(zeros[-1], zeros[-1]))
        return zeros

    def add_leaf(self, data: str) -> str:
        """add a leaf and return new root."""
        leaf_hash = MerkleTree._hash_leaf(data)
        self._leaves.append(leaf_hash)

        current = leaf_hash
        idx = self._next_index

        for level in range(self._depth):
            if idx % 2 == 0:
                # left child: save and move up
                self._filled_subtrees[level] = current
                current = MerkleTree._hash(current, self._zero_hashes[level])
            else:
                # right child: combine with saved left
                current = MerkleTree._hash(self._filled_subtrees[level], current)
            idx = idx // 2

        self._next_index += 1
        self._root = current

        return self._root

    @property
    def root(self) -> str:
        return self._root

    @property
    def size(self) -> int:
        return len(self._leaves)
