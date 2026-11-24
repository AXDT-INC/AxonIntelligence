"""CPU-only, disposable controls for the live backend and its host boundary."""
import contextlib
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import runner as r
import live_backend as b
import live_package as package
import live_supervisor as s
import live_worker as w
from live_store import Ledger

GPU = ["GPU-00000000-0000-0000-0000-000000000001","GPU-00000000-0000-0000-0000-000000000002"]
RT = r.Runtime(*["a"*64]*5,(0,),tuple(GPU))
COMMON = json.loads(json.dumps(asdict(RT)))
JOB = {"item_id":"1"*32,"input_ids":[2,3],"rendered_sha256":"f"*64,"options":[[5],[6,7]]}
LONG = {**JOB, "item_id":"2"*32,"input_ids":[3]*(r.CONTEXT-r.NEW_TOKENS)}
ADAPTER = "5280cf5f30265038a6fee6aa57955b829d387b7d78fb9fe6afab86d65ac50978"


def plan(mode="bare", ticket="1", synthetic=True):
    return {"run_id":"a"*32,"slot_ticket":ticket*32,"mode":mode,"image":"a"*64,"common":COMMON,
            "uuids":GPU, "synthetic":synthetic, "jobs":deepcopy([JOB,LONG] if synthetic else [JOB])}


class FakeOOM(RuntimeError):
    pass


class MockModel:
    torch = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=FakeOOM, max_memory_allocated=lambda i:1,
                                                 max_memory_reserved=lambda i:2))
    runtime = RT

    def __init__(self, mode):
        self.mode = mode

    def decode(self, ids, limit):
        return r.Decode(r.digest(ids), (5,0), "toy", r.Measurement(1,.5,(1,2),(3,4),16))

    def start_aux(self): pass

    def aux_peaks(self): return {"allocated_bytes":[1,2],"reserved_bytes":[3,4],"cpu_rss_bytes":16}

    def probe(self, ids):
        return {"input_sha256":r.digest(ids), "argmax":5, "checked_context_tokens":len(ids)+(r.NEW_TOKENS if len(ids)==r.CONTEXT-r.NEW_TOKENS else 1), "terminal_logits_sha256":"b"*64, "cache_reference_max_abs":0,
                "logits_sha256": ("b" if self.mode == "adapter" else "a")*64}

    def option_logprobs(self, prefix, suffix):
        return [-.5]*len(suffix)


class MockChannel:
    def __init__(self, plan):
        self.plan = plan
        self.model = MockModel(plan["mode"])
        self.ops = []

    def exchange(self, request, timeout):
        if "mode" in request:
            identity = {"mode":request["mode"],"common":COMMON, "load_reserved_bytes":[1,2],
                        "warmup":{"measurement":{"peak_reserved_bytes":[2,3]}},"model":{
                "provenance_sha256":r.PROVENANCE_SHA256, "base_revision":r.BASE_REVISION,
                "adapter_sha256":ADAPTER if request["mode"] in {"adapter","disabled"} else None}}
            return {"ready":identity,"fixtures":self.plan["jobs"] if request["synthetic"] else []}
        self.ops.append(request["op"])
        value = w.execute(self.model,request["op"],self.plan["run_id"],self.plan["slot_ticket"], request["job"])
        return {"request_sha256":r.digest(request),"result":value}


class Controls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.key = self.root/"key"
        self.key.write_bytes(os.urandom(32))
        self.key.chmod(0o600)

    def ledger(self, p):
        directory = self.root/p["slot_ticket"]
        directory.mkdir(mode=0o700,exist_ok=True)
        db = Ledger(directory, self.key, s.session(r.digest(p),"a"*64))
        self.addCleanup(db.close)
        return db

    def run_block(self, p):
        s.validate_plan(p)
        db = self.ledger(p)
        ch = MockChannel(p)
        s.drive(p, db, lambda:contextlib.nullcontext(ch))
        return db, ch

    def test_separate_loads_blinded_audit_and_determinism(self):
        blocks = []
        for i, mode in enumerate(("bare","disabled","adapter","bare","adapter")):
            p = plan(mode,str(i))
            db,ch = self.run_block(p)
            self.assertEqual(ch.ops,["generate","options","probe"]*2)
            results = s.collect(p,db)
            self.assertEqual(results[JOB["item_id"]+"/options"]["value"][1]["log_probability"], -1)
            self.assertEqual(results[JOB["item_id"]+"/generate"]["tokens_per_second"],2)
            blocks.append((p,results))
            s.drive(p,db,lambda:self.fail("completed attempts replayed"))
        s.compare(blocks,True)
        blocks[1][1][JOB["item_id"]+"/probe"]["value"]["logits_sha256"]="f"*64
        with self.assertRaises(r.IntegrityError):
            s.compare(blocks,True)

    def test_pilot_pair_vs_missing_or_mutated_inputs(self):
        blocks=[]
        for i,m in enumerate(("bare","adapter")):
            p=plan(m,str(i),False)
            db,ch=self.run_block(p)
            self.assertEqual(ch.ops,["generate","options"])
            blocks.append((p,s.collect(p,db)))
        s.compare(blocks,False)
        blocks[1][1][JOB["item_id"]+"/generate"]["receipt"]["actual_input_sha256"]="d"*64
        with self.assertRaises(r.IntegrityError):
            s.compare(blocks,False)

    def test_timeout_is_terminal_and_cannot_be_rescored(self):
        p=plan()
        db=self.ledger(p)
        ch=MockChannel(p)
        exchange=ch.exchange
        ch.exchange=lambda req,t: exchange(req,t) if "mode" in req else (_ for _ in ()).throw(TimeoutError())
        with self.assertRaises(r.IntegrityError):
            s.drive(p,db,lambda:contextlib.nullcontext(ch))
        res=db.get("done/"+JOB["item_id"]+"/generate")["receipt"]
        self.assertEqual(res["status"],"timeout")
        self.assertIsNone(res["text"])
        with self.assertRaises(r.IntegrityError):
            s.drive(p,db,lambda:self.fail("replayed after timeout"))

    def test_interrupted_assignment_no_second_attempt(self):
        p=plan()
        db=self.ledger(p)
        name,task=next(s.ops(p))
        db.put("claim/"+name,r.digest(task))
        with self.assertRaises(r.IntegrityError):
            s.drive(p,db,lambda:self.fail("launched"))
        self.assertEqual(db.get("done/"+name)["receipt"]["status"],"worker_lost")

    def test_private_store_authentication_lock_and_transaction(self):
        p=plan()
        db=self.ledger(p)
        with self.assertRaises(BlockingIOError):
            Ledger(self.root/p["slot_ticket"],self.key,db.session)
        db.put("first",{"v":1})
        with self.assertRaises(sqlite3.IntegrityError):
            db.put_many({"new":123,"first":123})
        self.assertIsNone(db.get("new"))
        db.db.execute("UPDATE events SET payload=? WHERE id='first'",(b'{"v":0}',))
        with self.assertRaises(r.IntegrityError):
            db.get("first")

    def test_reject_grading_and_role_leaks(self):
        for field in ("key","score","reviewer","answer","dataset_path"):
            with self.assertRaises(r.IntegrityError):
                w.unpack({**JOB, field:"fixture"})
        with self.assertRaises(r.IntegrityError):
            s.validate_plan(plan("disabled",synthetic=False))
        with self.assertRaises(r.IntegrityError):
            w.parse(b'{"item":1,"item":2}')
        with self.assertRaises(r.IntegrityError):
            s.validate_plan({**plan(),"jobs":[JOB,JOB]})
        with self.assertRaises(r.IntegrityError):
            w.unpack({**JOB,"input_ids":[True]})
        with self.assertRaises(r.IntegrityError):
            w.unpack({**JOB,"options":[[2],[2]]})

    def test_plan_pin_runtime_and_file_permissions(self):
        p=plan()
        pin=s.save_plan(self.root,p)
        path=self.root/"plan.json"
        self.assertEqual(s.planspec(path,pin),p)
        with self.assertRaises(r.IntegrityError):
            s.planspec(path,"d"*64)
        with self.assertRaises(FileExistsError):
            s.save_plan(self.root,p)
        path.chmod(0o644)
        with self.assertRaises(r.IntegrityError):
            s.planspec(path,pin)
        with self.assertRaises(r.IntegrityError):
            s.validate_plan({**p,"uuids":list(reversed(GPU))})

    def test_loader_attestation_mismatch_stops_before_jobs(self):
        p=plan()
        db=self.ledger(p)
        ch=MockChannel({**p,"mode":"adapter"})
        ch.exchange=lambda req,t: {"ready":{"mode":"bare","common":COMMON,"model":{
            "base_revision":r.BASE_REVISION,"provenance_sha256":r.PROVENANCE_SHA256,
            "adapter_sha256":ADAPTER}}, "fixtures":[JOB,LONG]}
        with self.assertRaises(r.IntegrityError):
            s.drive(p,db,lambda:contextlib.nullcontext(ch))
        self.assertIsNone(db.get("claim/"+JOB["item_id"]+"/generate"))

    def test_oom_overflow_and_auxiliary_nonfinite(self):
        m=MockModel("bare")
        m.decode=Mock(side_effect=FakeOOM())
        call=lambda job:w.execute(m,"generate","1"*32,"2"*32,job)["receipt"]
        self.assertEqual(call(JOB)["status"],"oom")
        self.assertEqual(call({**JOB,"input_ids":[1]*4000})["status"],"context_overflow")
        self.assertEqual(m.decode.call_count,1)
        m.option_logprobs=lambda pre,suf:[float("nan")]*len(suf)
        self.assertEqual(w.execute(m,"options","1"*32,"2"*32,JOB)["status"],"nonfinite_logits")
        self.assertEqual(b.conditional_scores([1],[[2],[2,3]],lambda p,s:[-.25]*len(s)),
                         [{"tokens":1,"log_probability":-.25},{"tokens":2,"log_probability":-.5}])

    def test_real_pipe_deadline_cpu_child(self):
        proc=subprocess.Popen([sys.executable,"-c","import time;time.sleep(3)"],
                              stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        try:
            with self.assertRaises(TimeoutError):
                s.Channel(proc).exchange({"test":True},.03)
        finally:
            proc.kill()
            proc.wait()
            proc.stdin.close()
            proc.stdout.close()

    def test_package_and_container_custody(self):
        archive=self.root/"payload.tar"
        pins=package.build(archive)
        other=package.build(self.root/"repeat.tar")
        self.assertEqual(pins,other)
        root=self.root/"pkg"
        root.mkdir()
        with tarfile.open(archive) as tar:
            self.assertEqual(set(tar.getnames()),set(package.FILES)|{package.MANIFEST})
            for m in tar:
                (root/m.name).write_bytes(tar.extractfile(m).read())
        package.verify(root,pins["manifest_sha256"])
        base=self.root/"base"
        base.mkdir()
        for f in json.loads((root/"PROVENANCE.json").read_bytes())["base_files"]:
            if "/models/base/" in f["path"]:
                (base/Path(f["path"]).name).touch()
        cmd=s.container_args(plan(),root,base,None,"test")
        self.assertIn("--network=none",cmd)
        self.assertIn("--read-only",cmd)
        self.assertIn("--log-driver=none",cmd)
        self.assertFalse(any(str(self.key) in arg for arg in cmd))
        self.assertEqual(cmd.count("--mount"),2)
        (base/"extra").touch()
        with self.assertRaises(r.IntegrityError):
            s.container_args(plan(),root,base,None,"test")
        (root/"live_worker.py").write_text("modified")
        with self.assertRaises(r.IntegrityError):
            package.verify(root,pins["manifest_sha256"])

    def test_operator_commands_discover_prepare_five_loads_and_audit(self):
        import io
        archive=self.root/"block.tar"
        pkg=self.root/"package"
        pkg.mkdir()
        pin=package.build(archive)["manifest_sha256"]
        with tarfile.open(archive) as tar:
            for member in tar:
                (pkg/member.name).write_bytes(tar.extractfile(member).read())
        discovery=self.root/"discovery"
        blocks=self.root/"blocks"
        for root in (discovery,blocks):
            root.mkdir(mode=0o700)
        launches=[]
        @contextlib.contextmanager
        def mock_container(cmd,name,p,root,base,adapter):
            launches.append(p["mode"])
            channel=MockChannel(p)
            if p["mode"]=="discover":
                hello=channel.exchange
                channel.exchange=lambda req,t: {**hello(req,t),"fixtures":[JOB,LONG]}
            yield channel
        def cli(argv):
            with patch.object(sys,"argv",["live_supervisor.py",*map(str,argv)]), contextlib.redirect_stdout(io.StringIO()):
                s.main()
        def approved(root):
            return package.sha((root/"plan.json").read_bytes())
        baseargs=["--package-pin",pin,"--key",self.key]
        old_umask=os.umask(0o077)
        self.addCleanup(os.umask,old_umask)
        with patch.object(s,"__file__",str(pkg/"live_supervisor.py")), patch.object(s,"cleanup"), \
                patch.object(s,"container_args",return_value=[]), patch.object(s,"container",mock_container):
            cli(["init",discovery,"a"*64,*GPU])
            cli(["run",*baseargs,"--root",discovery,"--pin",approved(discovery),"--base","/fixture"])
            cli(["prepare",*baseargs,"--root",discovery,"--pin",approved(discovery),"--output",blocks])
            audit=["audit","--synthetic",*baseargs]
            for label in ("bare1","disabled","active1","bare2","active2"):
                root=blocks/label
                args=["run",*baseargs,"--root",root,"--pin",approved(root),"--base","/fixture"]
                if label.startswith(("active","disabled")):
                    args.extend(["--adapter","/fixture-adapter"])
                cli(args)
                audit.extend(["--block",root,approved(root)])
            cli(audit)
        self.assertEqual(launches,["discover","bare","disabled","adapter","bare","adapter"])


class DecodeTests(unittest.TestCase):
    def test_actual_decode_loop_cache_eos_and_cap(self):
        class Tensor:
            def __init__(self, data): self.data=data
            def __getitem__(self, idx): return self
            def tolist(self): return self.data
            def argmax(self): return self
            def item(self): return self.data

        for sequence,limit,expected in (([7,0,5],3,(7,0)),([9,8,6],2,(9,8))):
            m=b.Backend.__new__(b.Backend)
            m.torch=SimpleNamespace(manual_seed=lambda seed:None,cuda=SimpleNamespace(
                reset_peak_memory_stats=lambda i:None,max_memory_allocated=lambda i:1,max_memory_reserved=lambda i:2))
            m.eos=(0,)
            m.context=contextlib.nullcontext
            m.sync=lambda:None
            m.tensor=Tensor
            calls=[]
            def logits(**kw):
                calls.append(kw)
                return SimpleNamespace(logits=Tensor(sequence[len(calls)-1]),past_key_values="cache")
            m.logits=logits
            m.tokenizer=SimpleNamespace(decode=lambda *a,**kw:"toy")
            result=m.decode([3,4],limit)
            self.assertEqual(result.input_sha256,r.digest([3,4]))
            self.assertEqual(result.output_ids,expected)
            self.assertEqual(calls[1]["input_ids"].tolist(),[expected[0]])
            self.assertEqual(calls[1]["past_key_values"],"cache")
            result.measurement.validate()


class BackendBoundaries(unittest.TestCase):
    def test_teacher_forcing_positions_on_device(self):
        from unittest.mock import MagicMock
        backend = b.Backend.__new__(b.Backend)
        backend.context = contextlib.nullcontext
        backend.device = "input-device"
        backend.torch = MagicMock()
        forwarded = MagicMock()
        forwarded.__getitem__.return_value.tolist.return_value = [1,2,3]
        labels = MagicMock()
        backend.tensor = Mock(side_effect=[forwarded, labels])
        raw = MagicMock()
        raw.shape = (2,b.VOCAB)
        raw.device = "output-device"
        raw.log_softmax.return_value.gather.return_value.flatten.return_value.tolist.return_value = [-.25,-1.5]
        backend.logits = MagicMock()
        backend.logits.return_value.logits.__getitem__.return_value.float.return_value = raw
        self.assertEqual(backend.option_logprobs([1,2,3],[4,5]),[-.25,-1.5])
        self.assertEqual(backend.tensor.call_args_list[0].args,([1,2,3,4],))
        self.assertEqual(backend.tensor.call_args_list[1].args,([4,5],))
        backend.logits.assert_called_once_with(input_ids=forwarded,use_cache=False,logits_to_keep=2)
        labels.__getitem__.return_value.to.assert_called_once_with("output-device")
        raw.log_softmax.assert_called_once_with(-1)

    def test_disabled_context_restores_adapter_even_on_failure(self):
        import types
        events = []
        @contextlib.contextmanager
        def disable():
            events.append("off")
            try:
                yield
            finally:
                events.append("on")
        backend = b.Backend.__new__(b.Backend)
        backend.model = SimpleNamespace(disable_adapter=disable)
        backend.torch = SimpleNamespace(inference_mode=contextlib.nullcontext,
                                         autocast=Mock(return_value=contextlib.nullcontext()), bfloat16="bf16")
        mod=types.ModuleType("torch.nn.attention")
        mod.SDPBackend=SimpleNamespace(MATH="math")
        mod.sdpa_kernel=Mock(return_value=contextlib.nullcontext())
        with patch.dict(sys.modules,{"torch.nn.attention":mod}):
            for backend.mode in ("bare","adapter","disabled"):
                with self.assertRaises(RuntimeError):
                    with backend.context(): raise RuntimeError()
        self.assertEqual(events,["off","on"])
        self.assertEqual(mod.sdpa_kernel.call_args.args,(["math"],))

    def test_probe_exercises_4096_even_when_greedy_eos(self):
        from unittest.mock import MagicMock
        backend=b.Backend.__new__(b.Backend)
        backend.context=contextlib.nullcontext
        ids=(1,)*(r.CONTEXT-r.NEW_TOKENS)
        tensor=MagicMock()
        tensor.__getitem__.return_value.tolist.return_value=list(ids)
        backend.tensor=Mock(return_value=tensor)
        raw=MagicMock()
        raw.argmax.return_value.item.return_value=0
        raw.contiguous.return_value.cpu.return_value.numpy.return_value=b"mock-floats"
        step=MagicMock()
        step.logits.__getitem__.return_value.float.return_value=raw
        backend.logits=Mock(return_value=step)
        result=backend.probe(ids)
        self.assertEqual(backend.logits.call_count,1+r.NEW_TOKENS)
        self.assertEqual(result["checked_context_tokens"],4096)
        self.assertEqual(result["argmax"],0)
        self.assertEqual(backend.logits.call_args.kwargs["past_key_values"],step.past_key_values)

    def test_effective_container_rejects_unplanned_mount(self):
        from pathlib import Path
        p=plan()
        doc={"State":{"Status":"created"},"Image":"sha256:"+p["image"],"Config":{"User":"65534:65534"},
             "HostConfig":{"Privileged":False,"ReadonlyRootfs":True,"NetworkMode":"none","IpcMode":"private",
             "CapDrop":["ALL"],"SecurityOpt":["no-new-privileges:true"],"Memory":128*2**30,"MemorySwap":128*2**30,
             "PidsLimit":512,"LogConfig":{"Type":"none"},"DeviceRequests":[{"DeviceIDs":GPU}]},
             "Mounts":[{"Type":"bind","Source":"/code","Destination":"/opt/axworker","RW":False},
                       {"Type":"bind","Source":"/base","Destination":"/model/base","RW":False}]}
        check=lambda:s.inspect_container(doc,p,Path("/code"),Path("/base"),None)
        check()
        doc["Mounts"].append({"Type":"bind","Source":"/fixture","Destination":"/other","RW":False})
        with self.assertRaises(r.IntegrityError): check()
        doc["Mounts"].pop()
        doc["HostConfig"]["DeviceRequests"][0]["DeviceIDs"]=list(reversed(GPU))
        with self.assertRaises(r.IntegrityError): check()

    def test_memory_gate_preserves_primary_receipt(self):
        value = w.execute(MockModel("bare"),"generate","1"*32,"2"*32,JOB)
        value["receipt"]["measurement"]["peak_reserved_bytes"] = [s.PREFLIGHT_PEAK_BYTES+1,1]
        self.assertEqual(s.terminal_halt(plan(),{"op":"generate"},value),"synthetic_capacity_limit")
        self.assertIsNone(s.terminal_halt(plan(synthetic=False),{"op":"generate"},value))

    def test_identity_hashes_attached_base_values_and_actual_placement(self):
        class Parameter:
            dtype = "bf16"
            device = "cuda:0"
            shape = [2]
            def __init__(self, data): self.data = data
            def numel(self): return 26895998464
            def detach(self): return self
            def view(self,*_): return self
            def cpu(self): return self
            def numpy(self): return self.data
            def split(self,*_): return [self]
        cls = type("Qwen3_5ForCausalLM",(),{})
        model=cls()
        param=Parameter(b"original")
        torch=SimpleNamespace(bfloat16="bf16",uint8="uint8",device=lambda dev,n:f"{dev}:{n}")
        layout={"model.layers.0":0}
        model.named_parameters=lambda:[("model.layers.0.q_proj.weight",param)]
        bare=b.inspect_base(model,torch,layout)
        model.named_parameters=lambda:[("model.layers.0.q_proj.base_layer.weight",param),
                                      ("model.layers.0.q_proj.lora_A.default.weight",Parameter(b"adapter"))]
        self.assertEqual(b.inspect_base(model,torch,layout),bare)
        param.data=b"changed"
        self.assertNotEqual(b.inspect_base(model,torch,layout),bare)
        param.device="cpu"
        with self.assertRaises(r.IntegrityError):
            b.inspect_base(model,torch,layout)
