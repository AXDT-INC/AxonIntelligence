#!/usr/bin/env python3
"""AxonAI-v1 versioned, manually gated distributed LoRA worker.

Run through control.py; CPU imports/encoding/checkpoint verification do not load weights.
Frozen input files are never modified. COMPLETE.json is written only after every
rank has finished saving. An adapter export alone is not a resumable checkpoint.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from datetime import datetime, timezone

VERSION = "axonai_v1_lora_v1_20261009"

def approved_fsdp_config(config):
    # Transformers 5.16.1 defaults to v2 and explicitly passes it to Accelerate,
    # overriding FSDP_VERSION. Bind the approved FSDP1 architecture at that API.
    selected = dict(config['fsdp'])
    if selected.get('version', 1) != 1:
        raise ValueError('Only the approved FSDP1 backend is authorized')
    selected['version'] = 1
    # PEFT reads this independently of Accelerate's plugin configuration. Qwen's
    # inherited no-split list also names a vision block absent from the text model.
    os.environ['FSDP_TRANSFORMER_CLS_TO_WRAP'] = ','.join(selected['transformer_layer_cls_to_wrap'])
    return selected

def validate_production_config(config):
    required = {'mode','model_path','expected_model_revision','world_size','max_length','num_train_epochs','max_steps','per_device_train_batch_size','per_device_eval_batch_size','gradient_accumulation_steps','learning_rate','weight_decay','warmup_steps','lr_scheduler_type','save_steps','checkpoint_minutes','eval_steps','logging_steps','seed','fsdp','lora'}
    missing = sorted(required - set(config))
    if missing: raise ValueError('missing config fields: ' + ','.join(missing))
    if config['mode'] not in ('smoke','main','resume') or config['world_size'] != 8: raise ValueError('invalid mode/world_size')
    if config['mode']=='resume' and config['max_steps'] != 3: raise ValueError('resume requires max_steps=3')
    if config['mode']=='resume' and config['gradient_accumulation_steps'] != 4: raise ValueError('invalid accumulation')
    for key in ('r','alpha','dropout','target_modules'):
        if key not in config['lora']: raise ValueError('missing lora field: '+key)
    for key in ('cpu_ram_efficient_loading','sync_module_states','transformer_layer_cls_to_wrap'):
        if key not in config['fsdp']: raise ValueError('missing fsdp field: '+key)
    return True


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


EVALUATION_WAIVER_SHA256 = "548564741451d0cfa5c8a858a8319b2675b7ec71208422aa4f9b00bea7daa8fb"


def evaluation_policy(mode, validation_file, waiver_path, identity, train_file):
    """Recognize the existing exact author waiver; grant no rights or launch approval."""
    if mode != "main":
        if waiver_path:
            raise ValueError("Internal SFT waiver applies to main mode only")
        return "no"
    if validation_file:
        if waiver_path:
            raise ValueError("Validation and evaluation waiver are mutually exclusive")
        return "steps"
    if not waiver_path:
        raise ValueError("A frozen validation file or the recorded evaluation waiver is required")
    raw = Path(waiver_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != EVALUATION_WAIVER_SHA256:
        raise ValueError("Evaluation waiver differs from the recorded author decision")
    waiver = json.loads(raw)
    if identity.get("evaluation_waiver_sha256") != EVALUATION_WAIVER_SHA256:
        raise ValueError("Run identity does not bind the evaluation waiver")
    if (identity.get("dataset_manifest_sha256") != waiver["training_manifest_sha256"]
            or sha256(train_file) != waiver["training_jsonl_sha256"]):
        raise ValueError("Evaluation waiver does not cover this dataset")
    return "no"


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temp.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, sort_keys=True, default=str, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def encode_example(tokenizer, row, max_length):
    """Encode one canonical single-answer conversation; never silently truncate.

    The exact locally pinned chat template is retained, including its thinking
    syntax. Prompt and empty thinking prefix receive -100 loss labels. Token
    boundary equality is mandatory: no approximate character-length masking.
    """
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("Expected canonical messages with a user and final assistant answer")
    roles = [m.get("role") for m in messages]
    if roles not in (["user", "assistant"], ["system", "user", "assistant"]):
        raise ValueError(f"Expected one user/assistant turn, optional system; got {roles}")
    if any(not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages):
        raise ValueError("All message content must be nonempty strings")
    opts = dict(tokenize=False, enable_thinking=False, preserve_thinking=False)
    prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, **opts)
    full = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **opts)
    if not full.startswith(prompt):
        raise ValueError("Chat template full answer is not prefixed by the generation prompt")
    prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if ids[:len(prefix)] != prefix or len(ids) <= len(prefix):
        raise ValueError("Tokenizer boundary mismatch or no supervised answer tokens")
    if len(ids) > max_length:
        raise ValueError(f"Example has {len(ids)} tokens, exceeds max_length={max_length}; truncation forbidden")
    if tokenizer.eos_token_id not in ids[len(prefix):]:
        raise ValueError("Assistant target lacks the tokenizer EOS token")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(prefix) + ids[len(prefix):]}


def verify_checkpoint(path, bindings=None, world_size=None, *, manifest_name="COMPLETE.json"):
    """Validate all saved files against an atomic completion manifest."""
    path = Path(path).resolve()
    manifest = json.loads((path / manifest_name).read_text())
    if manifest.get("schema_version") != 1 or not manifest.get("files"):
        raise ValueError("Invalid completion manifest")
    if bindings is not None and manifest["bindings"] != bindings:
        raise ValueError("Checkpoint run/config/data/worker bindings differ")
    if world_size is not None and manifest["world_size"] != world_size:
        raise ValueError("Checkpoint world size differs; this launcher requires unchanged topology")
    for name, entry in manifest["files"].items():
        f = (path / name).resolve()
        if not f.is_relative_to(path) or not f.is_file():
            raise ValueError(f"Unsafe or missing checkpoint file: {name}")
        if f.stat().st_size != entry["bytes"] or sha256(f) != entry["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {name}")
    files = set(manifest["files"])
    if not {"trainer_state.json", "scheduler.pt"}.issubset(files):
        raise ValueError("Checkpoint lacks trainer/scheduler state")
    if manifest["world_size"] > 1:
        required = {f"rng_state_{rank}.pth" for rank in range(manifest["world_size"])}
    else:
        required = {"rng_state.pth"}
    if not required.issubset(files):
        raise ValueError("Checkpoint lacks per-rank RNG state")
    if manifest["format"] == "fsdp_sharded":
        for folder in ("pytorch_model_fsdp_0", "optimizer_0"):
            if f"{folder}/.metadata" not in files or not any(n.startswith(folder + "/") and n.endswith(".distcp") for n in files):
                raise ValueError(f"Checkpoint lacks distributed {folder} state")
    elif manifest["format"] == "cpu_test":
        if "optimizer.pt" not in files or not any(n.endswith("adapter_model.safetensors") for n in files):
            raise ValueError("CPU test checkpoint lacks optimizer/adapter state")
    else:
        raise ValueError("Unsupported checkpoint format")
    state = json.loads((path / "trainer_state.json").read_text())
    if state["global_step"] != manifest["global_step"]:
        raise ValueError("Trainer global step differs from completion manifest")
    return manifest


def create_callback(run_dir, bindings, checkpoint_minutes=15, cpu_test=False, pause_at_step=None):
    from transformers import TrainerCallback
    import torch
    import torch.distributed as dist
    run_dir = Path(run_dir)

    class DurableCallback(TrainerCallback):
        def __init__(self):
            self.last_save_time = time.monotonic()
            self.stop_requested = False
            self.signal_requested = False

        def signal_stop(self, *_):
            # Do not run collectives or I/O from a POSIX signal handler.
            self.signal_requested = True

        def on_log(self, args, state, control, logs=None, **kwargs):
            if args.device.type == "cuda":
                append_json(run_dir / f"gpu_memory_rank-{args.process_index}.jsonl", {
                    "time": now(), "global_step": state.global_step,
                    "allocated_bytes": torch.cuda.memory_allocated(args.device),
                    "reserved_bytes": torch.cuda.memory_reserved(args.device),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device)})
            if state.is_world_process_zero:
                append_json(run_dir / "metrics.jsonl", {"time": now(), "global_step": state.global_step, **(logs or {})})

        def on_step_end(self, args, state, control, **kwargs):
            stop = self.signal_requested or (run_dir / "STOP_REQUEST").exists() or (pause_at_step is not None and state.global_step >= pause_at_step)
            save = time.monotonic() - self.last_save_time >= checkpoint_minutes * 60
            flags = torch.tensor([int(stop), int(save)], device=args.device, dtype=torch.int32)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(flags, op=dist.ReduceOp.MAX)
            stop, save = map(bool, flags.tolist())
            if stop or save or state.global_step >= state.max_steps:
                control.should_save = True
            if stop:
                self.stop_requested = True
                control.should_training_stop = True
            return control

        def on_save(self, args, state, control, **kwargs):
            # Trainer has saved model, optimizer, scheduler, RNG, trainer state.
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            error = None
            if state.is_world_process_zero:
                try:
                    checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                    files = {}
                    for path in sorted(checkpoint.rglob("*")):
                        if path.is_file() and path.name not in ("COMPLETE.json", "COMPLETE.pending.json"):
                            files[str(path.relative_to(checkpoint))] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
                    manifest = {"schema_version": 1, "created_at": now(), "version": VERSION,
                                "format": "cpu_test" if cpu_test else "fsdp_sharded",
                                "global_step": state.global_step, "world_size": args.world_size,
                                "bindings": bindings, "files": files}
                    # Check required files before publishing completion marker.
                    required = {"trainer_state.json", "scheduler.pt"}
                    if not required.issubset(files):
                        raise ValueError("Trainer did not produce expected resumable state")
                    atomic_json(checkpoint / "COMPLETE.pending.json", manifest)
                    verify_checkpoint(checkpoint, bindings, args.world_size, manifest_name="COMPLETE.pending.json")
                    os.replace(checkpoint / "COMPLETE.pending.json", checkpoint / "COMPLETE.json")
                    atomic_json(run_dir / "LATEST_COMPLETE.json", {"checkpoint": str(checkpoint.resolve()), "global_step": state.global_step, "manifest_sha256": sha256(checkpoint / "COMPLETE.json")})
                    append_json(run_dir / "lifecycle.jsonl", {"event": "checkpoint_complete", "time": now(), "checkpoint": str(checkpoint), "global_step": state.global_step})
                except BaseException:
                    error = traceback.format_exc()
            if dist.is_available() and dist.is_initialized():
                errors = [error]
                dist.broadcast_object_list(errors, src=0)
                error = errors[0]
            if error:
                raise RuntimeError("Checkpoint completion validation failed:\n" + error)
            self.last_save_time = time.monotonic()
            return control

    return DurableCallback()


def run(args):
    from guard import verify_worker
    verify_worker(args)
    run_dir = Path(args.run_dir).resolve()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    validate_production_config(config)
    identity_path = run_dir / "identity.json"
    identity_hash = sha256(identity_path)
    identity = json.loads(identity_path.read_text())
    if identity["config_sha256"] != sha256(config_path) or identity["code"]["train.py"] != sha256(__file__):
        raise ValueError("Worker/config changed after the run was prepared")
    freeze = json.loads((run_dir / "freeze.json").read_text())
    for split, input_path in (("train", args.train_file), ("validation", args.validation_file)):
        if input_path is None:
            continue
        entry = freeze["splits"][split]
        if not input_path or Path(input_path).resolve() != Path(entry["path"]).resolve() or sha256(input_path) != entry["sha256"]:
            raise ValueError(f"{split} input differs from the prepared freeze")
    authorization = json.loads((run_dir / "LAUNCH_AUTHORIZATION.json").read_text())
    if authorization.get("mode") != config["mode"] or authorization.get("identity_sha256") != identity_hash:
        raise ValueError("Launch authorization does not bind this mode and immutable run identity")
    if config["mode"] not in ("smoke", "main", "resume"):
        raise ValueError("Only manually authorized smoke/main modes are supported")
    if config["mode"] in ("smoke", "resume") and not 1 <= config["max_steps"] <= 10:
        raise ValueError("Smoke training must be capped at 1..10 optimizer steps")
    eval_strategy = evaluation_policy(config["mode"], args.validation_file,
                                      getattr(args, "evaluation_waiver", None), identity, args.train_file)
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model
    if not torch.cuda.is_available():
        raise ValueError("Production worker requires authorized CUDA execution; use core_cpu_test.py for CPU verification")
    if int(os.environ.get("WORLD_SIZE", "1")) != config["world_size"] or config["world_size"] != 8:
        raise ValueError("This release requires an explicitly launched eight-process FSDP run")
    if (run_dir / "STOP_REQUEST").exists():
        raise ValueError("STOP_REQUEST still exists; archive it before starting/resuming")
    bindings = {"identity_sha256": identity_hash, "config_sha256": sha256(config_path),
                "train_sha256": sha256(args.train_file), "validation_sha256": sha256(args.validation_file) if args.validation_file else None,
                "worker_sha256": sha256(__file__)}
    if getattr(args, "evaluation_waiver", None):
        bindings["evaluation_waiver_sha256"] = EVALUATION_WAIVER_SHA256
    resume = str(Path(args.resume).resolve()) if args.resume else None
    if not resume and any((run_dir / "checkpoints").glob("checkpoint-*")):
        raise ValueError("Existing checkpoints require an explicit verified --resume")
    package_versions = {p: importlib.metadata.version(p) for p in ("torch", "transformers", "accelerate", "peft", "datasets", "safetensors")}
    from accelerate.utils import enable_fsdp_ram_efficient_loading
    enable_fsdp_ram_efficient_loading()
    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=config["num_train_epochs"], max_steps=config["max_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"], weight_decay=config["weight_decay"],
        # Transformers 5.16.1 exposes warmup_steps (not warmup_ratio); resolve the
        # ratio during preflight once the bounded training-step count is known.
        warmup_steps=config["warmup_steps"], lr_scheduler_type=config["lr_scheduler_type"],
        bf16=True, fp16=False, tf32=False, optim="adamw_torch", max_grad_norm=1.0,
        fsdp=True, fsdp_config=approved_fsdp_config(config), gradient_checkpointing=False,
        save_strategy="steps", save_steps=config["save_steps"], save_only_model=False,
        save_total_limit=None, eval_strategy=eval_strategy, eval_steps=config["eval_steps"],
        prediction_loss_only=True, eval_accumulation_steps=1,
        logging_strategy="steps", logging_steps=config["logging_steps"], logging_nan_inf_filter=False,
        report_to="none", push_to_hub=False, seed=config["seed"], data_seed=config["seed"],
        dataloader_num_workers=0, dataloader_drop_last=False, remove_unused_columns=False,
        ignore_data_skip=False, restore_callback_states_from_checkpoint=False,
        ddp_timeout=3600, disable_tqdm=True,
    )
    atomic_json(run_dir / f"rank_ready_{training_args.process_index}.json", {
        "time": now(), "pid": os.getpid(), "rank": training_args.process_index,
        "world_size": training_args.world_size, "device": str(training_args.device),
        "device_name": torch.cuda.get_device_name(training_args.device),
        "approval_sha256": identity["approval_sha256"], "stage": "distributed_runtime_initialized"})
    if resume:
        # TrainingArguments has initialized torch.distributed; hash checkpoint once.
        import torch.distributed as dist
        error = None
        if training_args.process_index == 0:
            try:
                verify_checkpoint(resume, bindings, config["world_size"])
            except BaseException:
                error = traceback.format_exc()
        errors = [error]
        dist.broadcast_object_list(errors, src=0)
        if errors[0]:
            raise ValueError("Resume validation failed:\n" + errors[0])
        if config["mode"] == "resume":
            saved_step = json.loads((Path(resume) / "trainer_state.json").read_text()).get("global_step")
            if saved_step != 2:
                raise ValueError(f"Resume test requires checkpoint global_step=2, got {saved_step}")
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        raise ValueError("Pinned tokenizer must define a pad token")

    def dataset(path):
        records = []
        with Path(path).open() as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    raise ValueError(f"Blank dataset line {path}:{line_number}")
                try:
                    records.append(encode_example(tokenizer, json.loads(line), config["max_length"]))
                except Exception as exc:
                    raise ValueError(f"Dataset error {path}:{line_number}: {exc}") from exc
        if not records:
            raise ValueError(f"Empty dataset: {path}")
        return Dataset.from_list(records)

    train_dataset = dataset(args.train_file)
    validation_dataset = dataset(args.validation_file) if args.validation_file else None
    # AutoModelForCausalLM extracts qwen3_5 text_config and ignores documented
    # vision/MTP weights. Never use the inference-only device_map='auto'.
    model = AutoModelForCausalLM.from_pretrained(config["model_path"], local_files_only=True,
                trust_remote_code=False, dtype=torch.bfloat16, attn_implementation="sdpa")
    if model.__class__.__name__ != "Qwen3_5ForCausalLM":
        raise ValueError(f"Unexpected target text model class: {model.__class__.__name__}")
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    lc = config["lora"]
    present = {name.rsplit(".", 1)[-1] for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    if set(lc["target_modules"]) - present:
        raise ValueError(f"Missing LoRA target modules: {set(lc['target_modules']) - present}")
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                              bias="none", task_type="CAUSAL_LM", target_modules=lc["target_modules"]))
    callback = create_callback(run_dir, bindings, config["checkpoint_minutes"], pause_at_step=None)
    signal.signal(signal.SIGTERM, callback.signal_stop)
    signal.signal(signal.SIGINT, callback.signal_stop)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=validation_dataset, processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8),
        callbacks=[callback])
    if trainer.accelerator.state.fsdp_plugin.fsdp_version != 1:
        raise ValueError("Approved main architecture requires FSDP1")
    class ApprovedScheduleCallback(__import__('transformers').TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if state.max_steps != 668 or args.warmup_steps != 21 or args.num_train_epochs != 2:
                raise ValueError("Trainer schedule differs from authorized 668 steps / 21 warmup / 2 epochs")
            atomic_json(run_dir / f"rank_training_{args.process_index}.json", {
                "time": now(), "pid": os.getpid(), "rank": args.process_index,
                "world_size": args.world_size, "fsdp_version": 1,
                "max_steps": state.max_steps, "warmup_steps": args.warmup_steps,
                "approval_sha256": identity["approval_sha256"]})
    trainer.add_callback(ApprovedScheduleCallback())
    if trainer.is_world_process_zero():
        append_json(run_dir / "lifecycle.jsonl", {"event": "train_started", "time": now(), "resume_from": resume,
            "bindings": bindings, "packages": package_versions, "mode": config["mode"],
            "train_examples": len(train_dataset), "validation_examples": len(validation_dataset) if validation_dataset is not None else 0,
            "global_step": json.loads((Path(resume) / "trainer_state.json").read_text())["global_step"] if resume else 0})
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.accelerator.wait_for_everyone()
    last_checkpoint = run_dir / "checkpoints" / f"checkpoint-{trainer.state.global_step}"
    # Last-step callback saves even when max_steps is not a save_steps multiple.
    if trainer.is_world_process_zero():
        verify_checkpoint(last_checkpoint, bindings, config["world_size"])
    trainer.accelerator.wait_for_everyone()
    # Export is additional to the resumable sharded checkpoint, never its replacement.
    if not callback.stop_requested:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        trainer.save_model(str(run_dir / "final_adapter"))
        trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        outcome = {"success": True, "time": now(), "global_step": trainer.state.global_step,
                   "resumed_from": resume, "stop_requested": callback.stop_requested,
                   "final_checkpoint": str(last_checkpoint), "mode": config["mode"],
                   "world_size": config["world_size"], "bindings": bindings, "metrics": result.metrics}
        atomic_json(run_dir / "TRAINING_RESULT.json", outcome)
        append_json(run_dir / "lifecycle.jsonl", {"event": "train_finished", **outcome})
    trainer.accelerator.wait_for_everyone()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file")
    parser.add_argument("--evaluation-waiver")
    parser.add_argument("--resume")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    # Existing directory/identity is created by the launcher, not implicitly here.
    if not run_dir.is_dir():
        parser.error("Run directory must be prepared first")
    try:
        run(args)
    except BaseException:
        error_dir = run_dir / "errors"
        error_dir.mkdir(exist_ok=True)
        rank = os.environ.get("RANK", "0")
        with (error_dir / f"rank-{rank}-{time.time_ns()}.log").open("x") as f:
            f.write(now() + "\n" + traceback.format_exc())
            f.flush()
            os.fsync(f.fileno())
        raise


if __name__ == "__main__":
    main()
