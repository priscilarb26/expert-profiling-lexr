"""Inferência dos adaptadores QLoRA apenas no split congelado de TESTE.

Esta parte é separado do treinamento e da avaliação SBERT; aceita outra variante Qwen
com base/adapter compatíveis e hiperparâmetros específicos informados na CLI.

Mantém o prompt de TESTE do próprio fine-tuning, distinto dos prompts
few-shot da pasta src/llm. Reutiliza APENAS o parser e o escore posicional
compartilhados em src/llm/parse_tags.py. O split é conferido contra o arquivo
congelado: NÃO envia qrels, target_tags nem respostas ao modelo.

Defaults do experimento aplicado no modelo:
Qwen3.5-4B: batch=4; seed por bloco; n_tags=30;
max_input_tokens=8192; max_new_tokens=1024; do_sample=True; T=0.7;
top_p=0.8; top_k=20; min_p=0; repetition_penalty=1.0;
enable_thinking=False; NF4/double quant/bf16.

Uso:
 python -m src.finetuning.finetuned_inference \\
   --data-dir results/finetuning/data \\
   --split-path results/finetuning/data/split_autores_seed42.json \\
   --model Qwen/Qwen3.5-4B \\
   --adapter results/finetuning/qwen3_5_4b/adapter_qlora_final \\
   --output results/finetuning/qwen3_5_4b/tags_brutas_ft_teste.json

Requer GPU CUDA e as mesmas dependências do treinamento.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterator, **kwargs):  
        return iterator

try:
    from .prepare_dataset import construir_prompt_chat, ler_jsonl, normalizar_tag, salvar_json
    from ..llm.parse_tags import construir_ranking_qwen, parsear_tags_da_resposta
except ImportError:   
    from src.finetuning.prepare_dataset import (  
        construir_prompt_chat, ler_jsonl, normalizar_tag, salvar_json,
    )
    from src.llm.parse_tags import construir_ranking_qwen, parsear_tags_da_resposta  


def remover_thinking(texto: str) -> str:
    if not texto:
        return ""
    texto = re.sub(r"<think>.*?</think>", " ", texto, flags=re.DOTALL | re.IGNORECASE)
    texto = re.sub(r"</?think>", " ", texto, flags=re.IGNORECASE)
    return texto.strip()


def parsear_resposta_finetuning(resposta: str, n_tags: int) -> list[str]:
    """Reutiliza parser geral; no fine-tuning corta JSON e fallback em n_tags."""
    parsed = parsear_tags_da_resposta(remover_thinking(resposta), n_tags_pedir=n_tags)
    return [normalizar_tag(tag) for tag in parsed if normalizar_tag(tag)][:n_tags]


def carregar_modelo_e_tokenizer(args: argparse.Namespace):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("Instale torch CUDA, transformers, peft e bitsandbytes.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Inferência de adapter quantizado requer GPU CUDA.")
    if args.compute_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU não suporta bf16; use --compute-dtype fp16 explicitamente.")
    dtype = torch.bfloat16 if args.compute_dtype == "bf16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # a geração em batch deve começar no último token
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, torch_dtype=dtype,
        device_map={"": 0}, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False)
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    return tokenizer, model, torch


def identidade(args: argparse.Namespace, test_ids: list[str]) -> dict:
    config = {
        "model": args.model,
        "adapter": str(args.adapter.resolve()),
        "seed": args.seed, "batch_size": args.batch_size,
        "n_tags": args.n_tags, "enable_thinking": args.enable_thinking,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "top_k": args.top_k if args.do_sample else None,
        "min_p": args.min_p if args.do_sample else None,
        "repetition_penalty": args.repetition_penalty,
        "split_test_sha256": hashlib.sha256(
            json.dumps(test_ids, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    }
    return config


def gerar_tags_batch(
    rows: list[dict], tokenizer: Any, model: Any, torch: Any,
    args: argparse.Namespace,
) -> list[str]:
    prompts = [
        construir_prompt_chat(
            tokenizer, row["profile_text"], treino=False,
            n_tags=args.n_tags, enable_thinking=args.enable_thinking,
        ) for row in rows
    ]
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=args.max_input_tokens, add_special_tokens=False,
    ).to(model.device)
    kwargs = dict(
        max_new_tokens=args.max_new_tokens, do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    if args.do_sample:
        kwargs.update(
            temperature=args.temperature, top_p=args.top_p,
            top_k=args.top_k, min_p=args.min_p,
        )
    with torch.no_grad():
        generated = model.generate(**inputs, **kwargs)
    prompt_len = inputs.input_ids.shape[1]
    responses = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
    del inputs, generated
    return responses


def run(args: argparse.Namespace) -> None:
    if not args.adapter.joinpath("adapter_config.json").exists():
        raise FileNotFoundError(f"Adapter PEFT não encontrado em {args.adapter}")
    if args.batch_size <= 0 or args.n_tags <= 0:
        raise ValueError("batch-size e n-tags precisam ser positivos.")
    test_path = args.data_dir / "dataset_teste_215_autores.jsonl"
    split = json.loads(args.split_path.read_text(encoding="utf-8"))
    test_ids = list(split["test"])
    rows = ler_jsonl(test_path)
    by_author = {row["author_id"]: row for row in rows}
    if len(test_ids) != args.expected_test or set(test_ids) != set(by_author):
        raise ValueError("Dataset e split de TESTE não correspondem ao benchmark congelado.")
    for row in rows:
        if not row.get("profile_text"):
            raise ValueError(f"Autor {row.get('author_id')} sem profile_text.")
    config = identidade(args, test_ids)
    checkpoint = args.checkpoint or args.output.with_name(args.output.stem + "_checkpoint.json")
    if checkpoint.resolve() == args.output.resolve():
        raise ValueError("Checkpoint e resultado final devem ter caminhos diferentes.")

    raw: dict[str, list[str]] = {}
    errors: list[list[str]] = []
    if checkpoint.exists() and not args.fresh:
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved.get("identidade") != config:
            raise ValueError(
                f"Checkpoint {checkpoint} pertence a outro modelo/configuração/teste. "
                "Forneça outro --checkpoint ou use --fresh explicitamente."
            )
        raw = saved.get("tags_brutas_llm", {})
        errors = saved.get("autores_com_erro", [])
        if args.retry_errors and errors:
            failed = {record[0] for record in errors}
            for author in failed:
                raw.pop(author, None)
            errors = []
        if not set(raw).issubset(set(test_ids)):
            raise ValueError("Checkpoint contém autores que não pertencem ao teste atual.")
        print(f"Retomando checkpoint: {len(raw)}/{len(test_ids)} autores concluídos.")
    tokenizer, model, torch = carregar_modelo_e_tokenizer(args)
    from transformers import set_seed

    since_checkpoint = 0
    n_batches = math.ceil(len(test_ids) / args.batch_size)
    for batch_idx, start in enumerate(tqdm(
        range(0, len(test_ids), args.batch_size), total=n_batches, desc="Inferência teste"
    )):
        block = test_ids[start: start + args.batch_size]
        pending = [a for a in block if a not in raw]
        if not pending:
            continue
        batch_rows = [{"profile_text": by_author[author]["profile_text"]} for author in pending]
        set_seed(args.seed + batch_idx)  # mesma política de seed por bloco do notebook
        try:
            responses = gerar_tags_batch(batch_rows, tokenizer, model, torch, args)
            if len(responses) != len(pending):
                raise RuntimeError("O número de respostas não corresponde ao batch.")
            for author, response in zip(pending, responses):
                raw[author] = parsear_resposta_finetuning(response, args.n_tags)
        except Exception as exc:
            for author in pending:
                errors.append([author, str(exc)])
                raw[author] = []   
            gc.collect()
            torch.cuda.empty_cache()
            print(f"ERRO no batch {batch_idx}: {exc}")
        since_checkpoint += len(pending)
        if since_checkpoint >= args.checkpoint_every_authors:
            salvar_json(checkpoint, {
                "identidade": config,
                "seed": args.seed, "batch_llm": args.batch_size,
                "tags_brutas_llm": raw, "autores_com_erro": errors,
            })
            print(f"Checkpoint: {len(raw)}/{len(test_ids)} autores.")
            since_checkpoint = 0

    salvar_json(checkpoint, {
        "identidade": config, "seed": args.seed, "batch_llm": args.batch_size,
        "tags_brutas_llm": raw, "autores_com_erro": errors,
    })
    ordered_raw = {a: raw.get(a, []) for a in test_ids}
    salvar_json(args.output, ordered_raw)
    if args.ranking_output:
        rankings = {
            a: construir_ranking_qwen(ordered_raw[a]) for a in test_ids
        }
        salvar_json(args.ranking_output, rankings)
    print(f"Resultado final: {len(ordered_raw)} autores em {args.output}")
    print(f"Autores com erro de inferência (tags=[]): {len(errors)}")
    if errors:
        print("ATENÇÃO: verifique autores_com_erro no checkpoint. "
              "Use --retry-errors para tentar novamente, não interprete [] como predição válida.")
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--ranking-output", type=Path, default=None)
    parser.add_argument("--expected-test", type=int, default=215)
    parser.add_argument("--compute-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-every-authors", type=int, default=12)
    parser.add_argument("--n-tags", type=int, default=30)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignora checkpoint anterior; sobrescreve-o nesta execução.")
    args = parser.parse_args()
    if args.checkpoint_every_authors <= 0:
        parser.error("checkpoint-every-authors deve ser positivo")
    if args.do_sample and args.temperature <= 0:
        parser.error("temperature deve ser positivo com amostragem")
    run(args)


if __name__ == "__main__":
    main()
