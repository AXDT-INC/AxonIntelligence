"""Device-free checks for benchmark job handling and V100 admission."""
import inspect
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from transformers import AutoTokenizer
import infer
import placement
import worker

FIRST = "GPU-00000000-0000-0000-0000-000000000001"
SECOND = "GPU-00000000-0000-0000-0000-000000000002"


class QualificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("/model", local_files_only=True, trust_remote_code=False)

    def test_neutral_input_and_rejections(self):
        resident = Mock(tokenizer=self.tokenizer)
        fixture = {"id": "fixture", "prompt": " x" * 2050, "max_new_tokens": 4}
        out = worker.response(resident, fixture)
        self.assertEqual(out["status"], "context_overflow")
        self.assertEqual(worker.response(resident, dict(fixture, prompt="<|im_start|>"))["status"], "invalid_prompt")
        for cap in (0, 257, True):
            with self.assertRaises(RuntimeError):
                worker.response(resident, dict(fixture, max_new_tokens=cap))
        resident.decode.assert_not_called()
        ids = worker.encode(self.tokenizer, "hello")
        rendered = self.tokenizer.decode(ids[0])
        self.assertIn(worker.SYSTEM, rendered)
        self.assertNotIn("You are AxonAI", rendered)

    def test_swapped_checkpoint_fails_before_loading(self):
        actual = worker.hashlib.sha256((Path("/model") / "SHA256_MANIFEST.json").read_bytes()).hexdigest()
        other = next(arm for arm, pin in worker.PINS.items() if pin != actual)
        with self.assertRaisesRegex(ValueError, "manifest digest mismatch"):
            worker.checkpoint(other)

    def test_production_loader_has_no_benchmark_controls(self):
        self.assertEqual(list(inspect.signature(placement.read_model).parameters), ["root"])
        self.assertNotEqual(infer.SYSTEM, worker.SYSTEM)
        self.assertIs(infer.read_model, placement.read_model)
        if worker.hashlib.sha256((Path("/model") / "SHA256_MANIFEST.json").read_bytes()).hexdigest() == worker.PINS["qwen"]:
            with self.assertRaisesRegex(ValueError, "manifest digest mismatch"):
                placement.read_model(Path("/model"))

    def test_visible_devices_order_and_capacity(self):
        ordered = [FIRST, SECOND]
        props = [SimpleNamespace(uuid=u, name="Tesla V100", major=7, minor=0) for u in ordered]
        env = {name: ",".join(ordered) for name in (
            "BENCH_RESERVED_UUIDS", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")}
        rows = [{"budget": 100}, {"budget": 100}]
        with patch.dict(os.environ, env), patch.object(worker.subprocess, "check_output",
                return_value="\n".join(ordered)), patch.object(worker.torch, "cuda") as cuda:
            cuda.device_count.return_value = 2
            cuda.get_device_properties.side_effect = props.__getitem__
            cuda.mem_get_info.return_value = (101, 200)
            self.assertEqual(worker.gpu_pair(rows), ordered)
            cuda.mem_get_info.return_value = (99, 200)
            with self.assertRaisesRegex(RuntimeError, "Insufficient"):
                worker.gpu_pair(rows)
            props.reverse()
            with self.assertRaisesRegex(RuntimeError, "order"):
                worker.gpu_pair(rows)


if __name__ == "__main__":
    unittest.main()
