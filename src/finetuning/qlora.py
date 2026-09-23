"""Treina adaptadores QLoRA para Expert Profiling em modelos Qwen compatíveis.

Dataset e split são gerados separadamente em prepare_dataset.py.

Características: 4bit NF4 + dupla quantização, LoRA, loss SOMENTE no alvo
(-100 no prompt), early stopping, retomada de checkpoint, melhor modelo pela
validação, logs, tabela e curva de loss. O benchmark de teste nunca é usado
pelo Trainer. Teste OOM e checagem explícita da loss são opcionais.

Uso:
 python -m src.finetuning.qlora \\
   --data-dir results/finetuning/data \\
   --model Qwen/Qwen3.5-4B \\
   --output-dir results/finetuning/qwen3_5_4b

Dependências conforme notebook: torch CUDA, transformers>=5.5.4,
peft>=0.19.1, accelerate, bitsandbytes, datasets, pandas e matplotlib.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    from .prepare_dataset import (
        construir_prompt_chat, ler_jsonl, preparar_dataset_trainer, salvar_json,
    )
except ImportError:
    from prepare_dataset import (  
        construir_prompt_chat, ler_jsonl, preparar_dataset_trainer, salvar_json,
    )

DEFAULT_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def carregar_dependencias():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
            DataCollatorForSeq2Seq, Trainer, TrainingArguments,
            EarlyStoppingCallback, set_seed,
        )
        from transformers.trainer_utils import get_last_checkpoint
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise RuntimeError(
            "Dependência de treinamento ausente. Instale as versões indicadas no notebook."
        ) from exc
    return locals()


def definir_seed(seed: int, torch: Any, set_seed: Any) -> None:
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def dtype_gpu(torch: Any, name: str):
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA 4-bit requer GPU CUDA; nenhuma foi encontrada.")
    if name == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU não suporta bf16. Use --compute-dtype fp16 explicitamente.")
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Dtype inválido: {name}")


def bnb_config(d: dict, dtype: Any):
    return d["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )


def verificar_alvos_lora(model: Any, targets: list[str], d: dict) -> None:
    """Evita aplicar configurações Qwen incompatíveis entre famílias."""
    names = Counter()
    for fullname, module in model.named_modules():
        kind = type(module).__name__
        if isinstance(module, d["nn"].Linear) or "Linear4bit" in kind or "Linear8bit" in kind:
            names[fullname.rsplit(".", 1)[-1]] += 1
    missing = sorted(set(targets) - set(names))
    print("Módulos lineares disponíveis:", dict(names))
    print("Alvos LoRA escolhidos:", targets)
    if missing:
        raise ValueError(
            f"Esses alvos LoRA não existem neste modelo: {missing}. "
            "Ajuste --lora-target-modules de acordo com a arquitetura."
        )


def carregar_modelo_novo(args: argparse.Namespace, d: dict, dtype: Any):
    model = d["AutoModelForCausalLM"].from_pretrained(
        args.model,
        quantization_config=bnb_config(d, dtype),
        torch_dtype=dtype,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.config.use_cache = False
    verificar_alvos_lora(model, args.lora_targets, d)
    model = d["prepare_model_for_kbit_training"](model, use_gradient_checkpointing=True)
    lora = d["LoraConfig"](
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_targets,
    )
    model = d["get_peft_model"](model, lora)
    created = sum(1 for name, _ in model.named_modules() if "lora_A" in name)
    if not created:
        raise RuntimeError("Nenhuma camada LoRA foi criada.")
    model.print_trainable_parameters()
    print("Camadas LoRA criadas:", created)
    return model


def usar_adapter_terminado(args: argparse.Namespace, dataset_signatures: dict) -> bool:
    marker = args.output_dir / "treino_concluido.json"
    adapter_cfg = args.output_dir / "adapter_qlora_final" / "adapter_config.json"
    if not args.reuse_adapter or not (marker.exists() and adapter_cfg.exists()):
        return False
    stored = json.loads(marker.read_text(encoding="utf-8"))
    if stored.get("base_model") != args.model:
        raise ValueError("Adapter salvo pertence a outro modelo base. Escolha outra output-dir.")
    if stored.get("dataset_signatures") != dataset_signatures:
        raise ValueError("Os datasets não coincidem com os do adapter salvo. Não reutilize este adapter.")
    print("Adapter final já treinado. Reutilizando:", adapter_cfg.parent)
    return True


def testar_loss_so_alvo(model: Any, train_dataset: Any, collator: Any, d: dict) -> None:
    """Mesma checagem da célula 10, sem atualizar parâmetros."""
    row = train_dataset[0]
    batch = collator([{
        key: row[key] for key in ("input_ids", "attention_mask", "labels")
    }])
    batch = {key: value.to(model.device) for key, value in batch.items()}
    model.eval()
    with d["torch"].no_grad():
        result = model(**batch)
        logits, labels = result.logits, batch["labels"]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        manual = d["F"].cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), ignore_index=-100,
        )
    print("Loss do modelo:", float(result.loss.detach().cpu()))
    print("Cross-Entropy apenas no alvo:", float(manual.detach().cpu()))
    print("Tokens de prompt mascarados:", int((labels == -100).sum().item()))
    print("Tokens do alvo:", int((labels != -100).sum().item()))
    model.train()
    del batch, result, logits, labels, shift_logits, shift_labels, manual
    gc.collect()
    d["torch"].cuda.empty_cache()


def testar_memoria_oom(
    args: argparse.Namespace, d: dict, dtype: Any, tokenizer: Any,
    train_ds: Any, val_ds: Any, collator: Any,
) -> None:
    """Teste OPCIONAL isolado de VRAM.

    Faz forward/backward na maior sequência train+val e gera com 4 cópias do
    maior prompt do conjunto de teste. NÃO usa respostas/qrels do teste.
    """
    torch = d["torch"]
    gb = 1024 ** 3
    total = torch.cuda.get_device_properties(0).total_memory / gb
    print(f"[OOM] VRAM total: {total:.2f} GB")
    model = carregar_modelo_novo(args, d, dtype)
    largest = max(list(train_ds) + list(val_ds), key=lambda row: row["total_tokens"])
    batch = collator([{k: largest[k] for k in ("input_ids", "attention_mask", "labels")}])
    batch = {k: v.to(model.device) for k, v in batch.items()}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    output = model(**batch)
    output.loss.backward()
    model.zero_grad(set_to_none=True)
    print("[OOM] Pico treino (GB):", round(torch.cuda.max_memory_allocated()/gb, 2))
    del output, batch
    gc.collect()
    torch.cuda.empty_cache()

    tests = ler_jsonl(args.data_dir / "dataset_teste_215_autores.jsonl")
    largest_test = max(tests, key=lambda row: len(row["profile_text"]))
    tokenizer.padding_side = "left"
    prompt = construir_prompt_chat(
        tokenizer, largest_test["profile_text"], treino=False,
        n_tags=args.n_tags, enable_thinking=args.enable_thinking,
    )
    inputs = tokenizer(
        [prompt] * args.batch_inference,
        return_tensors="pt", padding=True, truncation=True,
        max_length=args.max_input_tokens, add_special_tokens=False,
    ).to(model.device)
    model.eval()
    model.config.use_cache = True
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens_oom, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    print("[OOM] Pico inferência (GB):", round(torch.cuda.max_memory_allocated()/gb, 2))
    del inputs, model
    gc.collect()
    torch.cuda.empty_cache()
    tokenizer.padding_side = "right"


def tabela_loss(log_history: list[dict]):
    import numpy as np
    import pandas as pd
    logs = pd.DataFrame(log_history)
    if logs.empty:
        return pd.DataFrame(columns=["Step", "Training Loss", "Validation Loss"])
    train = (
        logs[logs["loss"].notna()][["step", "loss"]].copy()
        if "loss" in logs else pd.DataFrame(columns=["step", "loss"])
    )
    valid = (
        logs[logs["eval_loss"].notna()][["step", "eval_loss"]].copy()
        if "eval_loss" in logs else pd.DataFrame(columns=["step", "eval_loss"])
    )
    rows = []
    for _, ev in valid.iterrows():
        step = int(ev["step"])
        exact = train[train["step"] == step]
        prev = train[train["step"] <= step]
        train_loss = (float(exact.iloc[-1]["loss"]) if len(exact)
                      else float(prev.iloc[-1]["loss"]) if len(prev) else np.nan)
        rows.append({"Step": step, "Training Loss": train_loss,
                     "Validation Loss": float(ev["eval_loss"])})
    return pd.DataFrame(rows)


def salvar_logs_e_grafico(log_history: list[dict], output_dir: Path, *, plot: bool) -> None:
    import pandas as pd
    logs = pd.DataFrame(log_history)
    logs.to_csv(output_dir / "logs_treinamento.csv", index=False, encoding="utf-8-sig")
    summary = tabela_loss(log_history)
    summary.to_csv(output_dir / "tabela_loss.csv", index=False, encoding="utf-8-sig")
    if logs.empty or not plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    train = logs[logs["loss"].notna()] if "loss" in logs else pd.DataFrame()
    valid = logs[logs["eval_loss"].notna()] if "eval_loss" in logs else pd.DataFrame()
    if train.empty and valid.empty:
        return
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    if not train.empty:
        ax.plot(train["step"], train["loss"], label="Loss treino")
    if not valid.empty:
        ax.plot(valid["step"], valid["eval_loss"], label="Loss validação")
    ax.set(title="Curva de loss — Fine-tuning QLoRA", xlabel="Step", ylabel="Loss")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()
    (output_dir / "graficos").mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "graficos" / "curva_loss.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_steps % args.eval_steps:
        raise ValueError("Para load_best_model_at_end, save_steps deve ser múltiplo de eval_steps.")
    if args.train_micro_batch * args.grad_accumulation != 4:
        print("ATENÇÃO: batch efetivo diferente dos 4 utilizados no experimento original.")
    train_path = args.data_dir / "dataset_treino_1001_autores.jsonl"
    valid_path = args.data_dir / "dataset_validacao_215_autores.jsonl"
    train_records, valid_records = ler_jsonl(train_path), ler_jsonl(valid_path)
    if len(train_records) != args.expected_train or len(valid_records) != args.expected_valid:
        raise ValueError(
            f"Datasets diferentes do benchmark: train={len(train_records)} "
            f"e validation={len(valid_records)}."
        )
    train_authors = {x["author_id"] for x in train_records}
    valid_authors = {x["author_id"] for x in valid_records}
    if train_authors & valid_authors:
        raise ValueError("Vazamento: os conjuntos treino/validação compartilham autores.")
    split_path = args.split_path or (args.data_dir / "split_autores_seed42.json")
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split por autor não encontrado em {split_path}. "
            "Forneça --split-path se estiver usando o split oficial em outro diretório."
        )
    split = json.loads(split_path.read_text(encoding="utf-8"))
    test_path = args.data_dir / "dataset_teste_215_autores.jsonl"
    test_ids = {row["author_id"] for row in ler_jsonl(test_path)}
    if (train_authors != set(split["train"]) or
        valid_authors != set(split["validation"]) or
        test_ids != set(split["test"]) or
        train_authors & test_ids or valid_authors & test_ids):
        raise ValueError("Os arquivos treino/validação/teste divergem do split congelado.")
    dataset_signatures = {
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(valid_path.read_bytes()).hexdigest(),
    }
    if usar_adapter_terminado(args, dataset_signatures):
        return

    d = carregar_dependencias()
    torch = d["torch"]
    dtype = dtype_gpu(torch, args.compute_dtype)
    definir_seed(args.seed, torch, d["set_seed"])
    tokenizer = d["AutoTokenizer"].from_pretrained(
        args.model, use_fast=True, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ds_train = preparar_dataset_trainer(
        train_records, tokenizer, max_input_tokens=args.max_input_tokens,
        max_target_tokens=args.max_target_tokens, enable_thinking=args.enable_thinking,
    )
    ds_valid = preparar_dataset_trainer(
        valid_records, tokenizer, max_input_tokens=args.max_input_tokens,
        max_target_tokens=args.max_target_tokens, enable_thinking=args.enable_thinking,
    )
    cols = ("input_ids", "attention_mask", "labels")
    train_trainer = ds_train.remove_columns([c for c in ds_train.column_names if c not in cols])
    valid_trainer = ds_valid.remove_columns([c for c in ds_valid.column_names if c not in cols])
    collator = d["DataCollatorForSeq2Seq"](
        tokenizer=tokenizer, model=None, padding=True,
        label_pad_token_id=-100, pad_to_multiple_of=8, return_tensors="pt",
    )
    print("Tokens máximo no treino:", max(ds_train["total_tokens"]))
    print("Tokens truncados de entrada:", sum(x >= args.max_input_tokens for x in ds_train["input_tokens"]))
    print("Tokens truncados de alvo:", sum(x >= args.max_target_tokens for x in ds_train["target_tokens"]))
    if args.smoke_test_oom:
        testar_memoria_oom(args, d, dtype, tokenizer, ds_train, ds_valid, collator)

    model = carregar_modelo_novo(args, d, dtype)
    if args.check_loss:
        testar_loss_so_alvo(model, ds_train, collator, d)
    steps_per_epoch = math.ceil(len(ds_train) / (args.train_micro_batch * args.grad_accumulation))
    warmup = math.ceil(args.warmup_ratio * steps_per_epoch * args.epochs)
    print(f"Warmup: {warmup} steps; épocas: {args.epochs}")
    ckpt_dir = args.output_dir / "checkpoints_treino"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        output_dir=str(ckpt_dir), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_micro_batch,
        per_device_eval_batch_size=args.eval_micro_batch,
        gradient_accumulation_steps=args.grad_accumulation,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        warmup_steps=warmup, lr_scheduler_type=args.scheduler,
        logging_steps=args.logging_steps, logging_strategy="steps",
        save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        bf16=args.compute_dtype == "bf16", fp16=args.compute_dtype == "fp16",
        optim=args.optim, max_grad_norm=args.max_grad_norm,
        gradient_checkpointing=True, report_to="none",
        remove_unused_columns=False, seed=args.seed, data_seed=args.seed,
    )
    try:
        targs = d["TrainingArguments"](**kwargs, eval_strategy="steps", eval_steps=args.eval_steps)
    except TypeError:
        targs = d["TrainingArguments"](**kwargs, evaluation_strategy="steps", eval_steps=args.eval_steps)
    trainer_kwargs = dict(
        model=model, args=targs, train_dataset=train_trainer,
        eval_dataset=valid_trainer, data_collator=collator,
        callbacks=[d["EarlyStoppingCallback"](early_stopping_patience=args.patience)],
    )
    try:
        trainer = d["Trainer"](**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = d["Trainer"](**trainer_kwargs, tokenizer=tokenizer)
    last_checkpoint = d["get_last_checkpoint"](str(ckpt_dir))
    print("Retomando checkpoint:", last_checkpoint or "nenhum")
    result = trainer.train(resume_from_checkpoint=last_checkpoint)
    print(result)
    adapter_dir = args.output_dir / "adapter_qlora_final"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    salvar_json(args.output_dir / "treino_concluido.json", {
        "status": "concluido", "base_model": args.model, "seed": args.seed,
        "dataset_signatures": dataset_signatures,
        "global_step": int(trainer.state.global_step),
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric_eval_loss": trainer.state.best_metric,
        "batch_efetivo": args.train_micro_batch * args.grad_accumulation,
    })
    salvar_json(args.output_dir / "configuracoes_experimento.json", {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    })
    salvar_logs_e_grafico(trainer.state.log_history, args.output_dir, plot=not args.no_plot)
    print("Adapter final salvo em:", adapter_dir)
    print("Inferência deve ser executada separadamente em finetuned_inference.py.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, default=None,
                        help="Split oficial usado na preparação; por padrão dentro de data-dir.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-target-tokens", type=int, default=1024)
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument("--compute-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default=DEFAULT_TARGETS)
    parser.add_argument("--train-micro-batch", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--eval-micro-batch", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--optim", default="paged_adamw_8bit")
    parser.add_argument("--scheduler", default="linear")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--expected-train", type=int, default=1001)
    parser.add_argument("--expected-valid", type=int, default=215)
    parser.add_argument("--n-tags", type=int, default=30, help="Usado apenas no OOM teste")
    parser.add_argument("--batch-inference", type=int, default=4, help="Usado apenas no OOM teste")
    parser.add_argument("--max-new-tokens-oom", type=int, default=1024)
    parser.add_argument("--smoke-test-oom", action="store_true",
                        help="Teste opcional e isolado; pode demandar tempo e VRAM.")
    parser.add_argument("--check-loss", action="store_true",
                        help="Compara loss calculada com CE manual ignorando prompt.")
    parser.add_argument("--no-reuse-adapter", dest="reuse_adapter", action="store_false")
    parser.add_argument("--no-plot", action="store_true")
    parser.set_defaults(reuse_adapter=True)
    args = parser.parse_args()
    args.lora_targets = [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]
    if not args.lora_targets:
        parser.error("--lora-target-modules não pode ficar vazio.")
    run(args)


if __name__ == "__main__":
    main()
