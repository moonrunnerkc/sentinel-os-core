# Author: Bradley R. Kinnard
# tests for authenticated sync

import pytest
import time

from crypto.pq_signatures import generate_keypair, Signer, Algorithm
from interfaces.authenticated_sync import (
    AuthenticatedSync,
    SignedExport,
    SyncResult,
    NonceTracker,
    ReplayDetectedError,
    SignatureInvalidError,
)


class TestNonceTracker:
    """tests for replay prevention."""

    def test_new_nonce_not_replay(self):
        tracker = NonceTracker()
        assert not tracker.is_replay("nonce_1", time.time())

    def test_seen_nonce_is_replay(self):
        tracker = NonceTracker()
        nonce = "test_nonce"
        ts = time.time()

        tracker.mark_seen(nonce, ts)
        assert tracker.is_replay(nonce, ts)

    def test_old_timestamp_is_replay(self):
        tracker = NonceTracker(max_age_seconds=3600)  # 1 hour
        old_ts = time.time() - 7200  # 2 hours ago

        assert tracker.is_replay("any_nonce", old_ts)

    def test_lru_eviction(self):
        tracker = NonceTracker(max_nonces=3)

        tracker.mark_seen("n1", time.time())
        tracker.mark_seen("n2", time.time())
        tracker.mark_seen("n3", time.time())
        tracker.mark_seen("n4", time.time())  # should evict n1

        assert not tracker.is_replay("n1", time.time())  # evicted
        assert tracker.is_replay("n4", time.time())  # still there

    def test_persistence_roundtrip(self):
        tracker1 = NonceTracker()
        tracker1.mark_seen("persist_test", time.time())

        data = tracker1.to_dict()

        tracker2 = NonceTracker()
        tracker2.load_dict(data)

        assert tracker2.is_replay("persist_test", time.time())


class TestAuthenticatedSync:
    """tests for authenticated sync operations."""

    def test_export_produces_valid_signature(self):
        keypair = generate_keypair(Algorithm.ED25519, key_id="test_export")
        signer = Signer(keypair)
        sync = AuthenticatedSync(signer, keypair, seed=42)

        beliefs = [{"id": "b1", "content": "test"}]
        export = sync.export_beliefs(beliefs)

        assert export.signer_key_id == "test_export"
        assert export.algorithm == "ed25519"
        assert len(export.signature) > 0
        assert len(export.nonce) > 0

    def test_import_verifies_signature(self):
        keypair = generate_keypair(Algorithm.ED25519, key_id="test_import")
        signer = Signer(keypair)
        sync = AuthenticatedSync(signer, keypair, seed=42)

        beliefs = [{"id": "b1", "content": "test"}]
        export = sync.export_beliefs(beliefs)

        result, imported = sync.import_beliefs(export)

        assert result.success
        assert result.imported_count == 1
        assert imported == beliefs

    def test_tampered_payload_rejected(self):
        keypair = generate_keypair(Algorithm.ED25519, key_id="test_tamper")
        signer = Signer(keypair)
        sync = AuthenticatedSync(signer, keypair, seed=42)

        beliefs = [{"id": "b1", "content": "test"}]
        export = sync.export_beliefs(beliefs)

        # tamper with payload
        export.payload["beliefs"][0]["content"] = "tampered"

        with pytest.raises(SignatureInvalidError):
            sync.import_beliefs(export)

    def test_replay_rejected(self):
        keypair = generate_keypair(Algorithm.ED25519, key_id="test_replay")
        signer = Signer(keypair)
        sync = AuthenticatedSync(signer, keypair, seed=42)

        beliefs = [{"id": "b1", "content": "test"}]
        export = sync.export_beliefs(beliefs)

        # first import succeeds
        result1, _ = sync.import_beliefs(export)
        assert result1.success

        # replay rejected
        with pytest.raises(ReplayDetectedError):
            sync.import_beliefs(export)

    def test_unknown_signer_rejected(self):
        keypair1 = generate_keypair(Algorithm.ED25519, key_id="sender")
        keypair2 = generate_keypair(Algorithm.ED25519, key_id="receiver")

        signer1 = Signer(keypair1)
        sync1 = AuthenticatedSync(signer1, keypair1, seed=42)

        signer2 = Signer(keypair2)
        sync2 = AuthenticatedSync(signer2, keypair2, seed=43)

        # sender exports
        beliefs = [{"id": "b1"}]
        export = sync1.export_beliefs(beliefs)

        # receiver doesn't know sender's key
        with pytest.raises(SignatureInvalidError) as exc:
            sync2.import_beliefs(export)
        assert "unknown signer" in str(exc.value)

    def test_cross_device_sync(self):
        """test realistic cross-device scenario."""
        # device A
        keypair_a = generate_keypair(Algorithm.ED25519, key_id="device_a")
        signer_a = Signer(keypair_a)
        sync_a = AuthenticatedSync(signer_a, keypair_a, seed=42)

        # device B
        keypair_b = generate_keypair(Algorithm.ED25519, key_id="device_b")
        signer_b = Signer(keypair_b)
        sync_b = AuthenticatedSync(signer_b, keypair_b, seed=43)

        # register each other's keys
        sync_a.register_peer_key(keypair_b)
        sync_b.register_peer_key(keypair_a)

        # A exports beliefs
        beliefs_from_a = [
            {"id": "b1", "content": "belief from A", "confidence": 0.8},
            {"id": "b2", "content": "another from A", "confidence": 0.6},
        ]
        export = sync_a.export_beliefs(beliefs_from_a)

        # B imports
        result, imported = sync_b.import_beliefs(export)

        assert result.success
        assert result.imported_count == 2
        assert imported == beliefs_from_a

    def test_serialization_roundtrip(self):
        keypair = generate_keypair(Algorithm.ED25519, key_id="test_serial")
        signer = Signer(keypair)
        sync = AuthenticatedSync(signer, keypair, seed=42)

        beliefs = [{"id": "b1"}]
        export = sync.export_beliefs(beliefs)

        # serialize to dict and back
        data = export.to_dict()
        restored = SignedExport.from_dict(data)

        result, imported = sync.import_beliefs(restored)
        assert result.success

    def test_determinism_under_seed(self):
        keypair = generate_keypair(Algorithm.ED25519, key_id="test_det")
        signer = Signer(keypair)

        sync1 = AuthenticatedSync(signer, keypair, seed=42)
        sync2 = AuthenticatedSync(signer, keypair, seed=42)

        beliefs = [{"id": "b1"}]

        export1 = sync1.export_beliefs(beliefs)
        export2 = sync2.export_beliefs(beliefs)

        # nonces should match under same seed
        assert export1.nonce == export2.nonce
