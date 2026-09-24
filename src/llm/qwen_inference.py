"""Inferência genérica das variantes Qwen sobre perfis já preparados do LExR.

São reutilizados os módulos ``prompts.py`` e ``parse_tags.py``. 
O nome do modelo e os hiperparâmetros são informados por argumentos,
sem criar um script separado para cada família.

Entrada: JSON de ``src/preprocessing/build_profiles_qwen.py``, contendo
``{id_autor: [{"titulo": ..., "keywords": ..., "abstract": ...}, ...]}``.
Saída: JSON de tags BRUTAS por autor. Opcionalmente, JSON dos rankings com
escores posicionais, calculados a partir das tags brutas.

O arquivo não executa SBERT, matching ou métricas. Para replicar o experimento
original, utilize o modo few-shot e os valores padrão deste script; passe
explicitamente os valores específicos dos outros experimentos.

A seleção de publicações ocorre por ``[:max_publicacoes]`` como no notebook,
SEM ordenar por ano. Verifique a ordem dos perfis antes da inferência.

Uso:
    python -m src.llm.qwen_inference \
        --profiles data/processed/perfis_estruturados_qwen.json \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --output results/tags_brutas_qwen2_5_1_5b.json \
        --ranking-output results/ranking_qwen2_5_1_5b.json

Dependências da inferência: pip install torch transformers accelerate tqdm
(usar uma versão do Transformers compatível com o modelo selecionado).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, **kwargs):  # type: ignore[no-redef]
        return it

try:
    from .prompts import montar_mensagens
    from .parse_tags import construir_ranking_qwen, parsear_tags_da_resposta
except ImportError:
    from prompts import montar_mensagens
    from parse_tags import construir_ranking_qwen, parsear_tags_da_resposta


@dataclass(frozen=True)
class GenerationConfig:
    #  O nome do modelo e os valores das variáveis variam de acordo com a família utilizada.
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    mode: str = "few-shot"
    n_tags: int = 30
    max_publicacoes: int = 50
    batch_size: int = 4
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float | None = None
    enable_thinking: bool | None = None
    repetition_penalty: float = 1.1
    max_new_tokens: int = 1024
    max_input_tokens: int = 16384
    seed: int = 42
    checkpoint_every_authors: int = 12

    def validate(self) -> None:
        if self.mode not in {"few-shot", "zero-shot"}:
            raise ValueError("mode deve ser few-shot ou zero-shot")
        for name in ("n_tags", "max_publicacoes", "batch_size", "max_new_tokens",
                     "max_input_tokens", "checkpoint_every_authors"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} deve ser positivo")
        if self.do_sample and not 0 < self.temperature:
            raise ValueError("temperature deve ser > 0 com do_sample=True")
        if self.do_sample and not 0 < self.top_p <= 1:
            raise ValueError("top_p deve pertencer a (0, 1] com do_sample=True")
        if self.min_p is not None and not 0 <= self.min_p <= 1:
            raise ValueError("min_p deve pertencer a [0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty deve ser > 0")

    def checkpoint_identity(self) -> dict[str, Any]:
        """Impede reaproveitar acidentalmente um checkpoint de outro modelo."""
        return {
            "model_name": self.model_name,
            "mode": self.mode,
            "n_tags": self.n_tags,
            "max_publicacoes": self.max_publicacoes,
            "do_sample": self.do_sample,
            "temperature": self.temperature if self.do_sample else None,
            "top_p": self.top_p if self.do_sample else None,
            "top_k": self.top_k if self.do_sample else None,
            "min_p": self.min_p if self.do_sample else None,
            "enable_thinking": self.enable_thinking,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
            "max_input_tokens": self.max_input_tokens,
            "seed": self.seed,
        }


def carregar_perfis_estruturados(caminho: Path) -> dict[str, list[dict[str, str]]]:
    """Lê perfis na mesma estrutura produzida por build_profiles_qwen.py."""
    with caminho.open("r", encoding="utf-8") as f:
        perfis = json.load(f)
    if not isinstance(perfis, dict):
        raise ValueError("O arquivo de perfis deve conter {id_autor: lista_de_publicacoes}.")
    for autor, publicacoes in perfis.items():
        if not isinstance(publicacoes, list) or not all(isinstance(p, dict) for p in publicacoes):
            raise ValueError(f"Perfil inválido do autor {autor!r}: esperado lista de objetos.")
    return perfis


def carregar_modelo_qwen(config: GenerationConfig):
    """Carrega o tokenizer e o modelo especificados"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">> Carregando {config.model_name} em {device}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # Essencial para que a geração em batch comece no último token do prompt.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    return tokenizer, model, device


def extrair_tags_do_llm_batch(
    tokenizer,
    model,
    device: str,
    publicacoes_lista: list[list[dict[str, str]]],
    config: GenerationConfig,
) -> list[list[str]]:
    """Gera uma lista de tags por autor, preservando a ordem do batch."""
    import torch

    prompts: list[str] = []
    for pubs in publicacoes_lista:
        mensagens = montar_mensagens(
            pubs, n_tags=config.n_tags,
            max_publicacoes=config.max_publicacoes, modo=config.mode,
        )
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if config.enable_thinking is not None:
            template_kwargs["enable_thinking"] = config.enable_thinking
        prompts.append(tokenizer.apply_chat_template(mensagens, **template_kwargs))
    # Reproduz padding esquerdo e truncamento originais do notebook.
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_input_tokens,
    ).to(device)

    kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
        "repetition_penalty": config.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if config.do_sample:
        kwargs.update(
            temperature=config.temperature, top_p=config.top_p, top_k=config.top_k,
        )
        if config.min_p is not None:
            kwargs["min_p"] = config.min_p
    # Qaundo do_sample = True valores são aplicados em temperature/top_p/top_k;
    # quando do_sample=False essas variáveis não são consideradas
    with torch.no_grad():
        outputs = model.generate(**inputs, **kwargs)

    prompt_len = inputs.input_ids.shape[1]
    resultados = []
    for i in range(outputs.shape[0]):
        resposta_tokens = outputs[i][prompt_len:]
        resposta = tokenizer.decode(resposta_tokens, skip_special_tokens=True).strip()
        resultados.append(parsear_tags_da_resposta(resposta, n_tags_pedir=config.n_tags))
    return resultados


def salvar_json_atomico(dados: Any, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_name(destino.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(dados, f, ensure_ascii=False)
        os.replace(tmp, destino)
    finally:
        tmp.unlink(missing_ok=True)


def salvar_checkpoint(
    caminho: Path,
    tags_brutas: dict[str, list[str]],
    autores_com_erro: list[tuple[str, str]],
    config: GenerationConfig,
) -> None:
    salvar_json_atomico({
        "config": config.checkpoint_identity(),
        "tags_brutas_llm": tags_brutas,
        "autores_com_erro": autores_com_erro,
    }, caminho)


def carregar_checkpoint(
    caminho: Path, config: GenerationConfig,
) -> tuple[dict[str, list[str]], list[tuple[str, str]]]:
    with caminho.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("config") != config.checkpoint_identity():
        raise ValueError(
            f"Checkpoint incompatível com os parâmetros atuais: {caminho}. "
            "Use um caminho de checkpoint diferente para cada execução/modelo."
        )
    tags = payload.get("tags_brutas_llm", {})
    erros = [tuple(item) for item in payload.get("autores_com_erro", [])]
    if not isinstance(tags, dict):
        raise ValueError("Checkpoint inválido: tags_brutas_llm não é objeto JSON.")
    return tags, erros


def executar_inferencia(
    perfis: dict[str, list[dict[str, str]]],
    config: GenerationConfig,
    *,
    caminho_saida: Path,
    caminho_checkpoint: Path,
    caminho_ranking: Path | None = None,
    caminho_erros: Path | None = None,
    manter_checkpoint: bool = False,
) -> tuple[dict[str, list[str]], dict[str, list[tuple[str, int]]]]:
    """Executa geração em lotes com checkpoint; não efetua avaliação semântica."""
    config.validate()
    try:
        import torch
        from transformers import set_seed
    except ImportError as e:
        raise RuntimeError(
            "A inferência requer torch e transformers: "
            "pip install torch transformers accelerate tqdm"
        ) from e

    tags_brutas: dict[str, list[str]] = {}
    erros: list[tuple[str, str]] = []
    if caminho_checkpoint.is_file():
        tags_brutas, erros = carregar_checkpoint(caminho_checkpoint, config)
        print(f">> Checkpoint recuperado: {len(tags_brutas)} autores já processados.")

    autores = list(perfis)
    desconhecidos = set(tags_brutas) - set(perfis)
    if desconhecidos:
        raise ValueError(
            "Checkpoint contém autores ausentes dos perfis atuais. "
            "Verifique se está retomando o mesmo experimento."
        )

    # A seed é fixada ANTES da primeira chamada generate.
    # Retomar um checkpoint com do_sample=True não reproduz necessariamente o
    # mesmo estado de RNG da execução contínua; tags já salvas não são refeitas.
    set_seed(config.seed)
    tokenizer, modelo, device = carregar_modelo_qwen(config)
    print(
        f">> Inferência {config.model_name}: autores={len(autores)}, "
        f"batch={config.batch_size}, modo={config.mode}, N={config.n_tags}"
    )
    desde_checkpoint = 0
    total_batches = math.ceil(len(autores) / config.batch_size)
    for i in tqdm(range(0, len(autores), config.batch_size),
                  total=total_batches, desc="Qwen batches"):
        lote = [a for a in autores[i:i + config.batch_size] if a not in tags_brutas]
        if not lote:
            continue
        pubs = [perfis[a] for a in lote]
        try:
            listas_tags = extrair_tags_do_llm_batch(tokenizer, modelo, device, pubs, config)
        except Exception as e:  
            for autor in lote:
                erros.append((autor, str(e)))
                tags_brutas[autor] = []
            print(f"[Aviso] Falha em lote ({', '.join(lote[:3])}...): {e}", file=sys.stderr)
            if device == "cuda":
                torch.cuda.empty_cache()
        else:
            if len(listas_tags) != len(lote):
                raise RuntimeError("A geração retornou quantidade inesperada de listas.")
            for autor, tags in zip(lote, listas_tags):
                tags_brutas[autor] = tags

        desde_checkpoint += len(lote)
        if desde_checkpoint >= config.checkpoint_every_authors:
            salvar_checkpoint(caminho_checkpoint, tags_brutas, erros, config)
            desde_checkpoint = 0
    if desde_checkpoint:
        salvar_checkpoint(caminho_checkpoint, tags_brutas, erros, config)

    salvar_json_atomico(tags_brutas, caminho_saida)
    ranking = {autor: construir_ranking_qwen(tags) for autor, tags in tags_brutas.items()}
    if caminho_ranking is not None:
        # JSON não admite tuplas; cada posição é armazenada como [tag, escore].
        salvar_json_atomico(ranking, caminho_ranking)
    if erros:
        destino_erros = caminho_erros or caminho_saida.with_name(
            caminho_saida.stem + "_erros.json"
        )
        salvar_json_atomico(erros, destino_erros)
        print(f"[Aviso] {len(erros)} autores com erro; detalhes em {destino_erros}")
    if not manter_checkpoint and not erros:
        caminho_checkpoint.unlink(missing_ok=True)
    del modelo, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f">> Tags brutas salvas em {caminho_saida} ({len(tags_brutas)} autores).")
    return tags_brutas, ranking


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True,
                        help="JSON produzido por build_profiles_qwen.py")
    parser.add_argument("--model", required=True,
                        help="Identificador Hugging Face, por exemplo Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", type=Path, required=True,
                        help="JSON de saída com tags brutas por autor")
    parser.add_argument("--ranking-output", type=Path,
                        help="JSON opcional com [[tag, escore_posicional], ...] por autor")
    parser.add_argument("--errors-output", type=Path,
                        help="JSON opcional dos autores cujos lotes falharam")
    parser.add_argument("--checkpoint", type=Path,
                        help="Checkpoint próprio do modelo (padrão: sufixo _checkpoint.json em --output)")
    parser.add_argument("--keep-checkpoint", action="store_true")
    parser.add_argument("--mode", choices=("few-shot", "zero-shot"), default="few-shot")
    parser.add_argument("--n-tags", type=int, default=30)
    parser.add_argument("--max-publicacoes", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True,
                        help="Padrão: ligado, como no notebook; use --no-do-sample para greedy decoding")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=None)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--enable-thinking", dest="enable_thinking", action="store_true")
    thinking.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.set_defaults(enable_thinking=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-input-tokens", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=12)
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint or args.output.with_name(
        args.output.stem + "_checkpoint.json"
    )
    destinos = [p.resolve() for p in (
        args.output, checkpoint, args.ranking_output, args.errors_output
    ) if p is not None]
    if len(destinos) != len(set(destinos)):
        parser.error("Arquivos de saída e checkpoint devem ter caminhos distintos.")
    if args.profiles.resolve() in destinos:
        parser.error("Nenhuma saída pode sobrescrever o arquivo de perfis.")

    config = GenerationConfig(
        model_name=args.model, mode=args.mode, n_tags=args.n_tags,
        max_publicacoes=args.max_publicacoes, batch_size=args.batch_size,
        do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k, repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens, max_input_tokens=args.max_input_tokens,
        seed=args.seed, checkpoint_every_authors=args.checkpoint_every,
    )
    perfis = carregar_perfis_estruturados(args.profiles)
    executar_inferencia(
        perfis, config, caminho_saida=args.output, caminho_checkpoint=checkpoint,
        caminho_ranking=args.ranking_output, caminho_erros=args.errors_output,
        manter_checkpoint=args.keep_checkpoint,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
