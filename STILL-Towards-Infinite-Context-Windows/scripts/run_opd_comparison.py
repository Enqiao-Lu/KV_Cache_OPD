"""Matched full-KV, STILL-style KL and KV OPD comparisons; default is a smoke."""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from still.core import StillCompactor
from still.data.qasper import load_documents
from still.train.opd import (
    OPDConfig,
    compare_teachers,
    document_loss,
    evaluate,
    generate_teacher_answers,
    gradient_stats,
)
from still.train.still import _build_training_schedule, _set_training_seed

MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


def parameter_digest(module) -> str:
    """Hash every parameter, including frozen backbone weights, one tensor at a time."""
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def select_documents(path, limit, question_limit):
    rows = load_documents(path)[:limit]
    for row in rows:
        row["questions"] = row["questions"][:question_limit]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, default=Path("outputs/opd/data/train.jsonl"))
    parser.add_argument("--eval-data", type=Path, default=Path("outputs/opd/data/dev.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/opd/smoke_4b"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--teacher-context", choices=["full", "evidence", "both"], default="both"
    )
    selection.add_argument(
        "--methods",
        nargs="+",
        choices=["still", "full", "evidence"],
        help="still: fixed full-teacher forward KL; full/evidence: student-on-policy JSD",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--num-latents", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-documents", type=int, default=1)
    parser.add_argument("--eval-documents", type=int, default=1)
    parser.add_argument("--questions-per-document", type=int, default=2)
    parser.add_argument("--max-source-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--tiny", action="store_true", help="random tiny Qwen3, real tokenizer")
    args = parser.parse_args()
    modes = args.methods or (
        ["full", "evidence"] if args.teacher_context == "both" else [args.teacher_context]
    )
    if len(modes) != len(set(modes)):
        parser.error("methods must be unique")
    if (
        min(
            args.num_latents,
            args.steps,
            args.train_documents,
            args.eval_documents,
            args.questions_per_document,
        )
        < 1
        or args.learning_rate <= 0
    ):
        parser.error("counts and learning rate must be positive")
    config = OPDConfig(
        max_source_tokens=args.max_source_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    training = select_documents(args.train_data, args.train_documents, args.questions_per_document)
    held_out = select_documents(args.eval_data, args.eval_documents, args.questions_per_document)
    if {r["document_id"] for r in training} & {r["document_id"] for r in held_out}:
        raise ValueError("Training/evaluation documents overlap")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    _set_training_seed(args.seed)
    if args.tiny:
        model = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=len(tokenizer),
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                max_position_embeddings=16384,
                eos_token_id=tokenizer.eos_token_id,
                attention_dropout=0.0,
            )
        )
        model.config._attn_implementation = args.attn_implementation
    else:
        model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
        )
    model.to(args.device).eval().requires_grad_(False)
    compactor = StillCompactor.from_model_config(model.config, num_latents=args.num_latents)
    compactor.to(args.device)
    initial = {n: tensor.detach().cpu().clone() for n, tensor in compactor.state_dict().items()}
    initial_digest = parameter_digest(compactor)
    backbone_digest = parameter_digest(model)
    metadata = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_path": snapshot,
        "tiny": args.tiny,
        "num_latents": args.num_latents,
        "seed": args.seed,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "methods": modes,
        "sampling_and_loss_config": asdict(config),
        "loss_by_method": {
            mode: (
                "full-vocabulary forward KL; fixed full-teacher answers; QASPER adaptation"
                if mode == "still"
                else "equal-weight full-vocabulary JSD; student-on-policy answers"
            )
            + "; token mean then question mean"
            for mode in modes
        },
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "attention_implementation": args.attn_implementation,
        "train_documents": [r["document_id"] for r in training],
        "eval_documents": [r["document_id"] for r in held_out],
        "train_data_sha256": hashlib.sha256(args.train_data.read_bytes()).hexdigest(),
        "eval_data_sha256": hashlib.sha256(args.eval_data.read_bytes()).hexdigest(),
        "command": sys.argv,
        "eos_token_ids": model.generation_config.eos_token_id,
        "backbone_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_compactor_parameters": sum(p.numel() for p in compactor.parameters()),
        "initial_compactor_sha256": initial_digest,
        "backbone_sha256": backbone_digest,
    }
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        metadata["gpu"] = torch.cuda.get_device_name(model.device)
    print(json.dumps(metadata, indent=2), flush=True)
    initial_eval = evaluate(model, tokenizer, compactor, held_out, config)
    if "evidence" in modes:
        _set_training_seed(args.seed)
        paired_teachers = compare_teachers(model, tokenizer, compactor, training[0], config)
        (args.output_dir / "teacher_diagnostics.json").write_text(
            json.dumps(paired_teachers, indent=2)
        )
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
        torch.cuda.synchronize(model.device)
    started = time.perf_counter()
    full_eval = evaluate(model, tokenizer, None, held_out, config)
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    results = {
        "full_context": {
            "evaluation": full_eval,
            "train_seconds": 0,
            "eval_seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(model.device) / 2**30
            if model.device.type == "cuda"
            else None,
        }
    }
    (args.output_dir / "full_context_result.json").write_text(
        json.dumps(results["full_context"], indent=2)
    )
    fixed_answers, teacher_preparation_seconds = {}, 0.0
    if "still" in modes:
        started = time.perf_counter()
        fixed_answers = {
            row["document_id"]: generate_teacher_answers(model, tokenizer, row, config)
            for row in training
        }
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        teacher_preparation_seconds = time.perf_counter() - started
        serialized = json.dumps(fixed_answers, sort_keys=True)
        metadata["fixed_teacher_answers_sha256"] = hashlib.sha256(serialized.encode()).hexdigest()
        (args.output_dir / "still_teacher_trajectories.json").write_text(
            json.dumps(
                {
                    "model_revision": MODEL_REVISION,
                    "backbone_sha256": backbone_digest,
                    "train_data_sha256": metadata["train_data_sha256"],
                    "max_new_tokens": config.max_new_tokens,
                    "greedy": True,
                    "answers_sha256": metadata["fixed_teacher_answers_sha256"],
                    "answers": fixed_answers,
                },
                indent=2,
            )
        )
    schedule = _build_training_schedule(len(training), args.steps, args.seed)
    for mode in modes:
        _set_training_seed(args.seed)
        compactor.load_state_dict(initial)
        assert parameter_digest(compactor) == initial_digest
        compactor.train()
        optimizer = torch.optim.AdamW(compactor.parameters(), lr=args.learning_rate)
        logs = []
        mode_config = replace(config, teacher_context="full" if mode == "still" else mode)
        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(model.device)
            torch.cuda.synchronize(model.device)
        started = time.perf_counter()
        for step, index in enumerate(schedule, 1):
            optimizer.zero_grad(set_to_none=True)
            row = training[index]
            loss, log = document_loss(
                model,
                tokenizer,
                compactor,
                row,
                mode_config,
                teacher_answers=fixed_answers[row["document_id"]] if mode == "still" else None,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"{mode}: nonfinite loss")
            loss.backward()
            log["gradient"] = gradient_stats(compactor)
            if not log["gradient"]["finite"] or log["gradient"]["norm"] <= 0:
                raise RuntimeError(f"{mode}: missing, zero or nonfinite compactor gradients")
            if log["gradient"]["layers_with_gradient"] != len(compactor.layers):
                raise RuntimeError(f"{mode}: some compactor layers have no gradient")
            if any(p.grad is not None or p.requires_grad for p in model.parameters()):
                raise RuntimeError("Backbone must remain frozen")
            torch.nn.utils.clip_grad_norm_(compactor.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            log["step"] = step
            logs.append(log)
            print(
                json.dumps(
                    {
                        "method": mode,
                        "step": step,
                        "loss": log["loss"],
                        "gradient": log["gradient"],
                        "source_tokens": log["source_tokens"],
                    }
                ),
                flush=True,
            )
            del loss
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        elapsed = time.perf_counter() - started
        updated_digest = parameter_digest(compactor)
        assert updated_digest != initial_digest, "optimizer did not change the compactor"
        assert parameter_digest(model) == backbone_digest, "frozen backbone weights changed"
        checkpoint_path = args.output_dir / f"{mode}_compactor.pt"
        torch.save(
            {
                "state_dict": {n: t.detach().cpu() for n, t in compactor.state_dict().items()},
                "metadata": metadata
                | {
                    "method": mode,
                    "teacher_context": mode_config.teacher_context,
                    "config": asdict(mode_config),
                },
                "optimizer": optimizer.state_dict(),
            },
            checkpoint_path,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        compactor.load_state_dict(checkpoint["state_dict"])
        assert parameter_digest(compactor) == updated_digest, "checkpoint round-trip differs"
        del checkpoint
        after = evaluate(model, tokenizer, compactor, held_out, config)
        results[mode] = {
            "objective": "forward_kl" if mode == "still" else "jsd",
            "loss_definition": metadata["loss_by_method"][mode],
            "initial_compactor_sha256": initial_digest,
            "initial_evaluation": initial_eval,
            "after_evaluation": after,
            "steps": logs,
            "train_seconds": elapsed,
            "teacher_preparation_seconds": teacher_preparation_seconds if mode == "still" else 0,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(model.device) / 2**30
            if model.device.type == "cuda"
            else None,
            "compactor_changed": True,
            "backbone_unchanged": True,
            "checkpoint_roundtrip": True,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "compactor_sha256": updated_digest,
        }
        (args.output_dir / f"{mode}_result.json").write_text(json.dumps(results[mode], indent=2))
        del optimizer
    if {"full", "evidence"}.issubset(modes):
        trajectories = [
            [q["answer_token_ids"] for q in results[m]["steps"][0]["questions"]]
            for m in ("full", "evidence")
        ]
        assert trajectories[0] == trajectories[1], "initial trajectories must match between arms"
        metadata["first_step_trajectories_matched"] = True
    summary = {
        "metadata": metadata,
        "results": results,
        "interpretation": (
            "Exploratory comparison; short runs do not establish method quality. "
            "STILL is a fixed full-teacher, full-vocabulary forward-KL QASPER adaptation. "
            "Different losses are not comparable quality scores."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# KV cache baseline comparison",
        "",
        summary["interpretation"],
        "",
        "| Method | Objective | Mean train loss | Gradient norm | "
        "Initial doc F1 | After doc F1 | Peak GiB |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        f"| full_context | none | — | — | — | {full_eval['document_f1']:.4f} | "
        f"{results['full_context']['peak_allocated_gib']} |",
    ]
    for mode in modes:
        result = results[mode]
        loss = sum(s["loss"] for s in result["steps"]) / len(result["steps"])
        grad = result["steps"][0]["gradient"]["norm"]
        lines.append(
            f"| {mode} | {result['objective']} | {loss:.6f} | {grad:.6f} | "
            f"{initial_eval['document_f1']:.4f} | "
            f"{result['after_evaluation']['document_f1']:.4f} | "
            f"{result['peak_allocated_gib']} |"
        )
    lines.extend(
        [
            "",
            "All trained arms: finite gradients, updated compactor, unchanged backbone, "
            "checkpoint round-trip checked. Gold answers are only read for evaluation.",
        ]
    )
    if "evidence" in modes:
        lines.extend(
            [
                "",
                "See `teacher_diagnostics.json` for full/evidence teacher answers, F1, "
                "and divergences on identical initial student trajectories.",
            ]
        )
    (args.output_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    print(f"Comparison saved: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
