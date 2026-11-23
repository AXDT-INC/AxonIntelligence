"""Host tests use headers and stdlib only. No parameter payload reads or CUDA imports."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from infer import uuids
from launch import check_reservation, run_args
from placement import GIB, plan, read_model

MODEL = Path("/data/axonai/models/merged/AxonAI-v1-27B-FP16")
TEST_IDS = ["GPU-00000000-0000-0000-0000-00000000000"+str(i) for i in range(2)]


class Candidate(unittest.TestCase):
    def test_every_weight_assigned(self):
        cfg, sizes = read_model(MODEL)
        self.assertEqual(len(sizes), 851)
        for count in (2,3,4):
            mapping, rows = plan(cfg, sizes, count)
            assigned = [0]*count
            for name, nbytes in sizes.items():
                owners = [i for k,i in mapping.items() if name.startswith(k+".")]
                self.assertEqual(len(owners),1)
                assigned[owners[0]] += nbytes
            self.assertEqual(assigned, [r["weights"] for r in rows])
            self.assertLess(max(assigned),27*GIB)
            self.assertEqual(sum(r["kv"]+r["recurrent"]+r["conv"] for r in rows),173801472)
            self.assertLess(max(r["budget"] for r in rows),32*GIB)

    def test_launch(self):
        csv = ",".join(TEST_IDS)
        with self.assertRaises(RuntimeError):
            uuids(",".join([TEST_IDS[0]]*2))
        command = run_args("sha256:"+"0"*64, uuids(csv), 123)
        self.assertIn('"device='+csv+'"', command)
        self.assertEqual(command.count("--mount"),1)
        self.assertIn("--read-only",command)
        self.assertIn("--network=none",command)
        with self.assertRaises(RuntimeError):
            run_args("latest",TEST_IDS,123)

    def test_claims(self):
        inventory = "\n".join(f"{i}, {uuid}" for i,uuid in enumerate(TEST_IDS))
        holder = json.dumps([{"State":{"Running": True}, "HostConfig":{"DeviceRequests":[{"DeviceIDs":["0","1"]}]}}])
        with patch("subprocess.check_output",side_effect=["0,1",inventory,holder,""]):
            check_reservation(123, TEST_IDS)
        for responses in ([""],["1,0",inventory], ["0,1",inventory,holder,TEST_IDS[1]]):
            with patch("subprocess.check_output",side_effect=responses),self.assertRaises(RuntimeError):
                check_reservation(123, TEST_IDS)
