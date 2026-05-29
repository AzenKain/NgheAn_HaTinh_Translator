#!/usr/bin/env python
"""Merge Whisper PEFT adapters and export browser-ready ONNX artifacts.

The output layout is compatible with Transformers.js:

    models/<model-name>/
        config.json
        generation_config.json
        tokenizer.json
        tokenizer_config.json
        preprocessor_config.json
        onnx/
            encoder_model.onnx
            decoder_model_merged.onnx
"""

from __future__ import annotations

import argparse
import functools
import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_ADAPTERS = (
    "whisper-lora-nghe-tinh",
    "whisper-dora-nghe-tinh",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge local Whisper LoRA/DoRA adapters and export ONNX for browser inference."
    )
    parser.add_argument(
        "--adapters",
        nargs="+",
        default=list(DEFAULT_ADAPTERS),
        help="PEFT adapter directories to export.",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to save exported ONNX models.",
    )
    parser.add_argument(
        "--work-dir",
        default="build/onnx-web",
        help="Temporary directory for merged PyTorch models.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Override base model. If omitted, uses adapter_config.json base_model_name_or_path.",
    )
    parser.add_argument(
        "--task",
        default="automatic-speech-recognition-with-past",
        help="Optimum ONNX task.",
    )
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset.")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device used by Optimum during export.",
    )
    parser.add_argument(
        "--dtype",
        default="fp32",
        choices=("fp32", "fp16", "bf16"),
        help="Floating point dtype for the exported base ONNX files.",
    )
    parser.add_argument(
        "--quantize",
        default="none",
        choices=("none", "int8"),
        help="Create dynamic int8 ONNX copies for smaller browser downloads.",
    )
    parser.add_argument(
        "--language",
        default="vi",
        help="Default Whisper language token saved in generation_config.json.",
    )
    parser.add_argument(
        "--whisper-task",
        default="transcribe",
        choices=("transcribe", "translate"),
        help="Default Whisper generation task.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Transformers and Optimum.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing merged/exported model directories.",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Reuse already merged models under --work-dir/merged.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip Optimum export and only refresh manifest/quantization checks.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def remove_dir(path: Path, force: bool) -> None:
    if not path.exists():
        return
    if not force:
        raise FileExistsError(f"{path} exists. Re-run with --force to overwrite it.")
    shutil.rmtree(path)


def ensure_preprocessor_config(source_dir: Path, target_dir: Path) -> None:
    """Write the Whisper feature-extractor config expected by Transformers.js."""
    target = target_dir / "preprocessor_config.json"
    if target.exists():
        return

    source = source_dir / "processor_config.json"
    if source.exists():
        processor_config = load_json(source)
        feature_config = processor_config.get("feature_extractor")
        if isinstance(feature_config, dict):
            feature_config = dict(feature_config)
            feature_config.setdefault(
                "processor_class", processor_config.get("processor_class", "WhisperProcessor")
            )
            write_json(target, feature_config)
            return

    fallback = {
        "chunk_length": 30,
        "feature_extractor_type": "WhisperFeatureExtractor",
        "feature_size": 80,
        "hop_length": 160,
        "n_fft": 400,
        "n_samples": 480000,
        "nb_max_frames": 3000,
        "padding_side": "right",
        "padding_value": 0.0,
        "processor_class": "WhisperProcessor",
        "return_attention_mask": False,
        "sampling_rate": 16000,
    }
    write_json(target, fallback)


def configure_generation(model: Any, processor: Any, language: str, whisper_task: str) -> None:
    model.config.use_cache = True
    model.config.forced_decoder_ids = None
    model.generation_config.language = language
    model.generation_config.task = whisper_task
    model.generation_config.forced_decoder_ids = None

    try:
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language,
            task=whisper_task,
            no_timestamps=True,
        )
    except Exception:
        return

    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.forced_decoder_ids = forced_decoder_ids


def merge_adapter(
    adapter_dir: Path,
    merged_dir: Path,
    base_model: str | None,
    language: str,
    whisper_task: str,
    trust_remote_code: bool,
    force: bool,
) -> str:
    try:
        import torch
        from peft import PeftModel
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Missing Python packages. Install them with: pip install -r requirements-onnx.txt"
        ) from exc

    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Missing PEFT config: {adapter_config_path}")

    adapter_config = load_json(adapter_config_path)
    resolved_base_model = base_model or adapter_config.get("base_model_name_or_path")
    if not resolved_base_model:
        raise ValueError(f"Cannot resolve base_model_name_or_path from {adapter_config_path}")

    remove_dir(merged_dir, force)
    merged_dir.mkdir(parents=True, exist_ok=True)

    print(f"[merge] {adapter_dir} -> {merged_dir}")
    print(f"[merge] base model: {resolved_base_model}")

    model = WhisperForConditionalGeneration.from_pretrained(
        resolved_base_model,
        dtype=torch.float32,
        trust_remote_code=trust_remote_code,
    )
    peft_model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()

    processor_source = adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else resolved_base_model
    try:
        processor = WhisperProcessor.from_pretrained(
            processor_source,
            language=language,
            task=whisper_task,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        print(
            f"[merge] Could not load processor from {processor_source}; "
            f"falling back to {resolved_base_model}. Reason: {exc}"
        )
        processor = WhisperProcessor.from_pretrained(
            resolved_base_model,
            language=language,
            task=whisper_task,
            trust_remote_code=trust_remote_code,
        )
    configure_generation(merged_model, processor, language, whisper_task)

    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    processor.save_pretrained(merged_dir)
    merged_model.generation_config.save_pretrained(merged_dir)
    ensure_preprocessor_config(adapter_dir, merged_dir)

    return resolved_base_model


def run_optimum_export(
    merged_dir: Path,
    export_dir: Path,
    task: str,
    opset: int,
    device: str,
    dtype: str,
    trust_remote_code: bool,
    force: bool,
) -> None:
    remove_dir(export_dir, force)
    export_dir.parent.mkdir(parents=True, exist_ok=True)

    command_preview = [
        sys.executable,
        "-m",
        "optimum.exporters.onnx",
        "--model",
        str(merged_dir),
        "--task",
        task,
        "--opset",
        str(opset),
        "--device",
        device,
        "--dtype",
        dtype,
        str(export_dir),
    ]
    if trust_remote_code:
        command_preview.insert(-1, "--trust-remote-code")

    print(f"[onnx] {' '.join(command_preview)}")
    export_with_optimum_api(
        merged_dir=merged_dir,
        export_dir=export_dir,
        task=task,
        opset=opset,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )


def quantize_int8(onnx_dir: Path) -> list[str]:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise RuntimeError(
            "Missing onnxruntime quantization package. Install with: pip install -r requirements-onnx.txt"
        ) from exc

    created: list[str] = []
    for model_path in sorted(onnx_dir.glob("*.onnx")):
        if model_path.stem.endswith("_int8"):
            continue
        quantized_path = model_path.with_name(f"{model_path.stem}_int8.onnx")
        print(f"[quantize] {model_path.name} -> {quantized_path.name}")
        quantize_dynamic(
            model_input=str(model_path),
            model_output=str(quantized_path),
            weight_type=QuantType.QInt8,
        )
        created.append(quantized_path.name)
    return created


def patch_optimum_partial_descriptors_for_py314() -> None:
    """Work around Python 3.14 binding functools.partial class attributes.

    Optimum stores some ONNX config factories as class-level functools.partial
    objects. On Python 3.14 those are returned as bound methods when accessed
    through an instance, which passes self as the first argument and breaks
    calls such as NORMALIZED_CONFIG_CLASS(config). Wrapping them with
    staticmethod preserves the intended factory behavior.
    """
    if sys.version_info < (3, 14):
        return

    import optimum.exporters.onnx.model_configs as model_configs

    for _, target_cls in vars(model_configs).items():
        if not inspect.isclass(target_cls):
            continue
        raw_value = target_cls.__dict__.get("NORMALIZED_CONFIG_CLASS")
        if isinstance(raw_value, functools.partial):
            setattr(target_cls, "NORMALIZED_CONFIG_CLASS", staticmethod(raw_value))


def export_with_optimum_api(
    merged_dir: Path,
    export_dir: Path,
    task: str,
    opset: int,
    device: str,
    dtype: str,
    trust_remote_code: bool,
) -> None:
    try:
        from optimum.exporters.onnx.__main__ import main_export
    except ImportError as exc:
        raise RuntimeError(
            "Missing optimum-onnx. Install it with: pip install -r requirements-onnx.txt"
        ) from exc

    patch_optimum_partial_descriptors_for_py314()
    main_export(
        model_name_or_path=str(merged_dir),
        output=export_dir,
        task=task,
        opset=opset,
        device=device,
        dtype=dtype,
        framework="pt",
        trust_remote_code=trust_remote_code,
    )


def validate_transformersjs_layout(model_dir: Path) -> None:
    required = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "onnx/encoder_model.onnx",
        "onnx/decoder_model_merged.onnx",
    )
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        detail = "\n".join(f"  - {name}" for name in missing)
        raise FileNotFoundError(
            f"{model_dir} is missing files required by Transformers.js:\n{detail}\n"
            "If decoder_model_merged.onnx is missing, upgrade optimum-onnx and export again."
        )


def normalize_transformersjs_layout(model_dir: Path) -> None:
    onnx_dir = model_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    for model_path in model_dir.glob("*.onnx"):
        target_path = onnx_dir / model_path.name
        if target_path.exists():
            target_path.unlink()
        shutil.move(str(model_path), str(target_path))


def detect_dtypes(onnx_dir: Path) -> list[str]:
    dtypes: list[str] = []
    if (onnx_dir / "encoder_model.onnx").exists() and (
        onnx_dir / "decoder_model_merged.onnx"
    ).exists():
        dtypes.append("fp32")
    if (onnx_dir / "encoder_model_int8.onnx").exists() and (
        onnx_dir / "decoder_model_merged_int8.onnx"
    ).exists():
        dtypes.append("int8")
    return dtypes


def refresh_manifest(output_dir: Path, models: list[dict[str, Any]]) -> None:
    manifest = {
        "sample_rate": 16000,
        "models": models,
    }
    write_json(output_dir / "model_manifest.json", manifest)


def main() -> int:
    args = parse_args()
    root = Path.cwd()
    output_dir = (root / args.output_dir).resolve()
    work_dir = (root / args.work_dir).resolve()
    merged_root = work_dir / "merged"

    exported_models: list[dict[str, Any]] = []
    for adapter_value in args.adapters:
        adapter_dir = (root / adapter_value).resolve()
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

        model_name = adapter_dir.name
        merged_dir = merged_root / model_name
        export_dir = output_dir / model_name

        base_model = args.base_model
        if not args.skip_merge:
            base_model = merge_adapter(
                adapter_dir=adapter_dir,
                merged_dir=merged_dir,
                base_model=args.base_model,
                language=args.language,
                whisper_task=args.whisper_task,
                trust_remote_code=args.trust_remote_code,
                force=args.force,
            )

        if not args.skip_export:
            run_optimum_export(
                merged_dir=merged_dir,
                export_dir=export_dir,
                task=args.task,
                opset=args.opset,
                device=args.device,
                dtype=args.dtype,
                trust_remote_code=args.trust_remote_code,
                force=args.force,
            )

        normalize_transformersjs_layout(export_dir)
        ensure_preprocessor_config(adapter_dir, export_dir)
        validate_transformersjs_layout(export_dir)

        if args.quantize == "int8":
            quantize_int8(export_dir / "onnx")

        dtypes = detect_dtypes(export_dir / "onnx")
        exported_models.append(
            {
                "id": model_name,
                "name": model_name,
                "base_model": base_model or load_json(adapter_dir / "adapter_config.json").get(
                    "base_model_name_or_path"
                ),
                "dtypes": dtypes,
                "default_dtype": "int8" if "int8" in dtypes else "fp32",
            }
        )

    refresh_manifest(output_dir, exported_models)
    print(f"[done] Browser models written to: {output_dir}")
    print(f"[done] Manifest: {output_dir / 'model_manifest.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"[error] Command failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)
