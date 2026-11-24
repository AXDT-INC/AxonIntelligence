import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from runner import (CONTEXT, NEW_TOKENS, Decode, IntegrityError, Item, ItemFailure, Measurement,
                    Runtime, audit_pairs, digest, render_item, run_item, sha, verify_directory,
                    verify_identity)


class SyntheticTokenizer:
    def apply_chat_template(self, messages, **options):
        self.render_options = options
        return "|".join(f"{x['role']}={x['content']}" for x in messages)

    def __call__(self, text, **options):
        self.encode_options = options
        return {"input_ids": list(text.encode("ascii"))}


class PairedRunnerTest(unittest.TestCase):
    def setUp(self):
        self.runtime = Runtime("1"*64, "2"*64, "3"*64, "4"*64, "5"*64, (0, 1),
                               ("GPU-00000000-0000-0000-0000-000000000001",
                                "GPU-00000000-0000-0000-0000-000000000002"))
        self.ticket, self.other, self.run = "a"*32, "b"*32, "c"*32
        self.tokenizer = SyntheticTokenizer()
        self.item = render_item("d"*32, "synthetic rectangle", self.tokenizer)
        self.metrics = Measurement(2.0, 0.5, (10, 12), (15, 16), 32)

    def decode(self, ids, limit):
        self.assertEqual(limit, NEW_TOKENS)
        return Decode(digest(ids), (18, 0), "shape", self.metrics)

    def one(self, ticket=None, item=None, generator=None):
        return run_item(self.run, ticket or self.ticket, item or self.item, self.runtime,
                        self.runtime, generator or self.decode)

    def audit(self, rows):
        return audit_pairs([self.item], rows, self.run, [self.ticket, self.other], self.runtime)

    def test_paired_round_trip_reproducible(self):
        rows = [self.one(), self.one(self.other)]
        self.assertEqual(rows[0], self.one())
        self.assertEqual(self.audit(json.loads(json.dumps(rows))), 2)
        self.assertEqual(rows[0]["input_sha256"], rows[1]["actual_input_sha256"])
        self.assertEqual(rows[0]["output_tokens"], 2)
        self.assertEqual(self.tokenizer.render_options,
                         {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False})
        self.assertEqual(self.tokenizer.encode_options,
                         {"add_special_tokens": False, "truncation": False, "padding": False})

    def test_delayed_response_may_differ(self):
        def second(ids, limit):
            return replace(self.decode(ids, limit), output_ids=(54, 11, 1), text="different")
        self.assertEqual(self.audit([self.one(), self.one(self.other, generator=second)]), 2)

    def test_runtime_or_model_input_drift_is_fatal(self):
        with self.assertRaises(IntegrityError):
            run_item(self.run, self.ticket, self.item, self.runtime,
                     replace(self.runtime, device_map_sha256="9"*64), self.decode)
        with self.assertRaises(IntegrityError):
            self.one(generator=lambda ids, limit: self.decode(ids[:-1], limit))

    def test_overflow_keeps_both_denominators(self):
        item = Item(self.item.item_id, self.item.rendered_sha256, (3,) * (CONTEXT - NEW_TOKENS + 1))
        def forbidden(*args):
            self.fail("overflow reached decoder")
        rows = [self.one(ticket, item, forbidden) for ticket in (self.ticket, self.other)]
        self.assertEqual(rows[0]["status"], "context_overflow")
        self.assertIsNone(rows[0]["output_tokens"])
        self.assertEqual(audit_pairs([item], rows, self.run, [self.ticket, self.other], self.runtime), 2)
        fit = replace(item, input_ids=item.input_ids[:-1])
        self.assertEqual(self.one(item=fit)["status"], "ok")

    def test_backend_failure_is_retained_but_integrity_error_propagates(self):
        def fail(ids, limit):
            raise ItemFailure("oom", replace(self.metrics, first_token_s=None))
        row = self.one(generator=fail)
        self.assertEqual(row["status"], "oom")
        self.assertIsNone(row["text"])
        self.assertEqual(self.audit([row, self.one(self.other)]), 2)
        def unknown(ids, limit):
            raise RuntimeError("unvalidated worker state")
        with self.assertRaises(RuntimeError):
            self.one(generator=unknown)

    def test_end_conditions_and_measurements(self):
        def result(ids, limit, output, text, measurement=None):
            return replace(self.decode(ids, limit), output_ids=output, text=text,
                           measurement=measurement or self.metrics)
        for tokens, text, expected in [((18,)*NEW_TOKENS, "shape", "length_limit"),
                                       ((18,)*NEW_TOKENS, "   ", "empty_output"),
                                       ((1,), "", "empty_output"),
                                       ((2,)*(NEW_TOKENS-1)+(0,), "shape", "ok")]:
            row = self.one(generator=lambda *args: result(*args, tokens, text))
            self.assertEqual(row["status"], expected)
            self.assertEqual(row["stop_reason"], "eos" if tokens[-1] in (0, 1) else "length_limit")
            self.assertEqual(self.audit([row, self.one(self.other)]), 2)
        for illegal in [(2, 3), (0, 8, 1), (True, 0)]:
            with self.assertRaises(IntegrityError):
                self.one(generator=lambda *args: result(*args, illegal, "shape"))
        for invalid in [replace(self.metrics, latency_s=float("nan")),
                        replace(self.metrics, first_token_s=5.0),
                        replace(self.metrics, peak_reserved_bytes=(0, 0))]:
            with self.assertRaises(IntegrityError):
                self.one(generator=lambda *args: result(*args, (2, 0), "shape", invalid))

    def test_reconcile_missing_duplicate_or_mismatched_receipts(self):
        one, other = self.one(), self.one(self.other)
        for rows in [[one], [one, one], [one, {**other, "run_id": "e"*32}],
                     [one, {**other, "input_sha256": "0"*64}], [one, {**other, "text": "altered"}],
                     [one, {**other, "status": "oom"}], [one, {**other, "output_tokens": 3}],
                     [one, {**other, "generation_sha256": "0"*64}]]:
            with self.assertRaises(IntegrityError):
                self.audit(rows)

    def test_private_model_manifest_and_artifact_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "synthetic-fixture"
            bundle.write_bytes(b"cpu")
            entries = [{"path": "/synthetic/synthetic-fixture", "sha256": sha(b"cpu"), "bytes": 3}]
            verify_directory(root, entries)
            bundle.write_bytes(b"bad")
            with self.assertRaises(IntegrityError):
                verify_directory(root, entries)
            with self.assertRaises(IntegrityError):
                verify_identity(bundle, root)
            bundle.write_bytes(b"cpu")
            (root / "other").write_text("unexpected")
            with self.assertRaises(IntegrityError):
                verify_directory(root, entries)


    def test_custodian_loads_only_registered_role_mounts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, adapter = root / "base", root / "adapter"
            base.mkdir()
            adapter.mkdir()

            def entry(mount, prefix, name, content):
                (mount / name).write_bytes(content)
                return {"path": prefix + "/" + name, "sha256": sha(content), "bytes": len(content)}

            base_entries = [entry(base, "/data/axonai/models/base/Qwen3.8-27B",
                                  f"fixture-{n}.safetensors", bytes([n])) for n in range(18)]
            weights = entry(adapter, "/synthetic", "adapter_model.safetensors", b"weights")
            config = entry(adapter, "/synthetic", "adapter_config.json", b"config")
            from runner import BASE_REVISION
            doc = {"base_revision": BASE_REVISION, "base_files": base_entries + [weights, config],
                   "adapter": weights, "adapter_config": config}
            manifest = root / "fixture.json"
            manifest.write_text(json.dumps(doc))
            with patch("runner.PROVENANCE_SHA256", sha(manifest.read_bytes())):
                self.assertIsNone(verify_identity(manifest, base)["adapter_sha256"])
                self.assertEqual(verify_identity(manifest, base, adapter)["adapter_sha256"],
                                 weights["sha256"])
                (adapter / "adapter_model.safetensors").write_bytes(b"changed")
                with self.assertRaises(IntegrityError):
                    verify_identity(manifest, base, adapter)


if __name__ == "__main__":
    unittest.main()
