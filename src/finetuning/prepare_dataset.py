"""Prepara dados supervisionados do fine-tuning do LExR por AUTOR.

Independe do nome/família do Qwen, mas conserva o protocolo analisado:
- qrels: relevâncias 1/2/3; colisão após normalização de espaços -> maior peso;
- perfil: até 50 publicações na ORDEM DE ENTRADA, com título, keywords, resumo;
- alvo: TODAS as tags LExR, ordenadas 3 -> 2 -> 1 e desempate alfabético;
- split por autor, seed 42, 1001/215/215 e autor ilustrativo forçado no teste;
- labels de treinamento: -100 em TODOS os tokens do prompt.

Uso:
 python -m src.finetuning.prepare_dataset \\
   --documents data/processed/filtered_documents.json \\
   --qrels data/processed/ground_truth/LExR-prof-qrels_filtrado \\
   --output-dir results/finetuning/data

Requer scikit-learn e tqdm (opcional); datasets/transformers apenas quando
as funções de tokenização forem chamadas pelo módulo qlora.py.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import csv
import json
from pathlib import Path
import re
from typing import Any, Optional

from sklearn.model_selection import train_test_split

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterator, **kwargs): 
        return iterator

RE_HTML = re.compile(r"<[^>]+>")
RE_ENTIDADE_HTML = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[A-Za-z]+);")
RE_CARACTERES_RUIM = re.compile(r"[^\w\sÀ-ÖØ-öø-ÿ.,;:!?()\-/'\"&+%#@=]", re.UNICODE)
RE_ESPACOS = re.compile(r"\s+")

SYSTEM_PROMPT = (
    "Você é um sistema especialista em perfilamento acadêmico e extração de perfis de expertise. "
    "Sua tarefa é analisar as publicações de um autor e extrair suas áreas de pesquisa em forma de tags. "
    "Responda APENAS com JSON válido, sem markdown, sem explicações."
)


def limpar_texto_basico(texto: Any) -> str:
    if not texto:
        return ""
    texto = RE_HTML.sub(" ", str(texto))
    texto = RE_ENTIDADE_HTML.sub(" ", texto)
    texto = RE_CARACTERES_RUIM.sub(" ", texto)
    return RE_ESPACOS.sub(" ", texto).strip()


def normalizar_tag(texto: Any) -> str:
    """Como o notebook de fine-tuning: apenas normaliza whitespace."""
    if not texto:
        return ""
    return RE_ESPACOS.sub(" ", str(texto)).strip()


def salvar_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        json.dump(value, out, ensure_ascii=False, indent=2)


def salvar_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")


def ler_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def carregar_qrels(path: Path) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, str]]]:
    gt_norm: dict[str, dict[str, int]] = defaultdict(dict)
    gt_original: dict[str, dict[str, str]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            author, tag_orig = str(parts[0]).strip(), parts[2]
            try:
                weight = int(parts[3])
            except (ValueError, TypeError):
                continue
            if weight not in {1, 2, 3}:
                continue
            tag = normalizar_tag(tag_orig)
            if not author or not tag:
                continue
            if tag not in gt_norm[author]:
                gt_norm[author][tag] = weight
                gt_original[author][tag] = tag_orig
            elif weight > gt_norm[author][tag]:
                gt_norm[author][tag] = weight
    return dict(gt_norm), dict(gt_original)


def estruturar_publicacao(doc: dict) -> dict[str, str]:
    title = limpar_texto_basico(doc.get("title") or "")
    abstract = limpar_texto_basico(doc.get("abstract") or "")
    kws = doc.get("keywords") or []
    keywords = ", ".join(
        limpar_texto_basico(k)
        for k in kws if isinstance(k, str) and k.strip()
    )
    return {"titulo": title, "keywords": keywords, "abstract": abstract}


def construir_perfis_estruturados(
    documents_jsonl: Path, authors: set[str]
) -> dict[str, list[dict[str, str]]]:
    profiles: dict[str, list[dict[str, str]]] = defaultdict(list)
    with documents_jsonl.open("r", encoding="utf-8") as source:
        for line in tqdm(source, desc="Publicações"):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue  
            if not isinstance(doc, dict):
                continue
            doc_authors = [str(a) for a in (doc.get("authors") or [])]
            valid = [a for a in doc_authors if a in authors]
            if not valid:
                continue
            pub = estruturar_publicacao(doc)
            if not pub["titulo"] and not pub["abstract"] and not pub["keywords"]:
                continue
            for author in valid:
                profiles[author].append(pub)
    return dict(profiles)


def construir_profile_text(publicacoes: list[dict], limite: int = 50) -> str:
    blocks = []
    for i, pub in enumerate(publicacoes[:limite], start=1):
        lines = [f"PUBLICAÇÃO {i}"]
        if pub.get("titulo"):
            lines.append(f"Título: {pub['titulo']}")
        if pub.get("keywords"):
            lines.append(f"Palavras-chave: {pub['keywords']}")
        if pub.get("abstract"):
            lines.append(f"Resumo: {pub['abstract']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks).strip()


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 42
    train_ratio: float = 0.70
    valid_ratio: float = 0.15
    test_ratio: float = 0.15
    expected_authors: int = 1431
    expected_train: int = 1001
    expected_valid: int = 215
    expected_test: int = 215
    forced_test_author: str = "9350794843721516"


def validar_split(split: dict, current_authors: set[str], c: SplitConfig) -> bool:
    try:
        train, val, test = [set(split[name]) for name in ("train", "validation", "test")]
    except (KeyError, TypeError, ValueError):
        return False
    return (
        split.get("seed") == c.seed and
        len(train) == c.expected_train and
        len(val) == c.expected_valid and
        len(test) == c.expected_test and
        not (train & val or train & test or val & test) and
        train | val | test == current_authors and
        c.forced_test_author in test
    )


def criar_split_por_autor(
    records: list[dict], split_path: Path, config: SplitConfig,
    *, rebuild: bool = False,
) -> dict:
    authors = sorted(rec["author_id"] for rec in records)
    all_authors = set(authors)
    if len(all_authors) != len(authors):
        raise ValueError("Há autores duplicados no dataset supervisionado.")
    if len(authors) != config.expected_authors:
        raise ValueError(f"Esperados {config.expected_authors} autores; recebidos {len(authors)}.")
    if config.forced_test_author not in all_authors:
        raise ValueError(f"Autor de exemplo {config.forced_test_author} ausente na base.")
    if abs(config.train_ratio + config.valid_ratio + config.test_ratio - 1) > 1e-12:
        raise ValueError("As frações train, validation e test precisam somar 1.")

    if split_path.exists() and not rebuild:
        existing = json.loads(split_path.read_text(encoding="utf-8"))
        if not validar_split(existing, all_authors, config):
            raise ValueError(
                f"Split existente inválido: {split_path}. Confira os IDs; "
                "use --rebuild-split somente se QUISER criar outro split."
            )
        print("Split existente validado e reutilizado:", split_path)
        return existing

    train, temp = train_test_split(
        authors, test_size=(config.valid_ratio + config.test_ratio),
        random_state=config.seed, shuffle=True,
    )
    valid, test = train_test_split(
        temp, test_size=config.test_ratio/(config.valid_ratio + config.test_ratio),
        random_state=config.seed, shuffle=True,
    )
    train, valid, test = list(train), list(valid), list(test)
    if config.forced_test_author not in test:
        substitute = sorted(test)[0]
        if config.forced_test_author in train:
            train.remove(config.forced_test_author)
            train.append(substitute)
        elif config.forced_test_author in valid:
            valid.remove(config.forced_test_author)
            valid.append(substitute)
        else:
            raise ValueError("Autor obrigatório não foi encontrado para troca.")
        test.remove(substitute)
        test.append(config.forced_test_author)
    split = {
        "seed": config.seed,
        "criterio": "split exclusivamente por author_id; publicações nunca são sorteadas separadamente",
        "forced_test_author": config.forced_test_author,
        "train": sorted(train),
        "validation": sorted(valid),
        "test": sorted(test),
    }
    if not validar_split(split, all_authors, config):
        raise AssertionError("Split gerado não passou na validação.")
    salvar_json(split_path, split)
    return split


def preparar_base_supervisionada(
    qrels: Path, documents: Path, output_dir: Path, *, max_publicacoes: int = 50,
    expected_authors: int = 1431, profiles_cache: Optional[Path] = None,
    rebuild_profiles: bool = False,
) -> list[dict]:
    gt, _ = carregar_qrels(qrels)
    authors = set(gt)
    cache = profiles_cache or (output_dir / "cache" / "perfis_estruturados_qwen_ft.json")
    if cache.exists() and not rebuild_profiles:
        profiles = json.loads(cache.read_text(encoding="utf-8"))
        print("Cache de perfis carregado:", cache)
    else:
        profiles = construir_perfis_estruturados(documents, authors)
        salvar_json(cache, profiles)
        print("Perfis estruturados gerados:", cache)

    records, minimal_profiles = [], []
    for author in sorted(authors):
        pubs = profiles.get(author, [])
        if not pubs:
            continue
        text = construir_profile_text(pubs, limite=max_publicacoes)
        if not text:
            continue
        info = sorted(
            [{"tag": tag, "relevance": int(weight)} for tag, weight in gt[author].items()],
            key=lambda x: (-x["relevance"], x["tag"]),
        )
        row = {
            "author_id": author,
            "profile_text": text,
            "n_publicacoes_usadas": min(len(pubs), max_publicacoes),
            "publicacoes": pubs[:max_publicacoes],
            "target_tags": [x["tag"] for x in info],
            "target_tags_info": info,
            "tags_relevancia_3": [x["tag"] for x in info if x["relevance"] == 3],
            "tags_relevancia_2": [x["tag"] for x in info if x["relevance"] == 2],
            "tags_relevancia_1": [x["tag"] for x in info if x["relevance"] == 1],
            "qrels": info,  
        }
        records.append(row)
        minimal_profiles.append({
            "author_id": author, "profile_text": text,
            "n_publicacoes_usadas": row["n_publicacoes_usadas"],
        })
    if len(records) != expected_authors:
        raise ValueError(
            f"Esperados {expected_authors} autores aproveitáveis, mas há {len(records)}. "
            "Verifique os arquivos e a filtragem antes de prosseguir."
        )
    salvar_jsonl(output_dir / "profile_text_qwen_ft.jsonl", minimal_profiles)
    salvar_jsonl(output_dir / "dataset_supervisionado_qwen_ft.jsonl", records)
    return records


def salvar_particoes(records: list[dict], split: dict, output_dir: Path) -> None:
    indexed = {row["author_id"]: row for row in records}
    for split_name, filename in [
        ("train", "dataset_treino_1001_autores.jsonl"),
        ("validation", "dataset_validacao_215_autores.jsonl"),
        ("test", "dataset_teste_215_autores.jsonl"),
    ]:
        salvar_jsonl(output_dir / filename, [indexed[a] for a in split[split_name]])
    with (output_dir / "ids_teste_215_autores.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["author_id"])
        writer.writerows([[a] for a in split["test"]])


def montar_instrucao_usuario(profile_text: str, treino: bool, n_tags: Optional[int] = None) -> str:
    if treino:
        instruction = (
            "Liste todas as tags de áreas de atuação sustentadas pelas publicações. "
            "Ordene da área mais central para a menos central. "
            "No alvo supervisionado, essa ordem corresponde às relevâncias LExR 3, depois 2, depois 1."
        )
    else:
        instruction = (
            f"Gere exatamente {n_tags} tags candidatas de áreas de atuação. "
            "Ordene da área mais central para a menos central."
        )
    return (
        f"Analise o perfil textual abaixo de um pesquisador.\n\n{instruction}\n\n"
        "Formato obrigatório de saída:\n"
        '{"tags": ["tag 1", "tag 2", "..."]}\n\n'
        f"Perfil textual do pesquisador:\n{profile_text}"
    )


def construir_prompt_chat(
    tokenizer: Any, profile_text: str, treino: bool, n_tags: Optional[int] = None,
    *, enable_thinking: bool = False,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": montar_instrucao_usuario(profile_text, treino=treino, n_tags=n_tags)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def montar_target_treino(tags_ordenadas: list[str]) -> str:
    return json.dumps({"tags": tags_ordenadas}, ensure_ascii=False)


def tokenizar_exemplo_treino(
    record: dict, tokenizer: Any, *, max_input_tokens: int = 8192,
    max_target_tokens: int = 1024, enable_thinking: bool = False,
) -> dict:
    prompt = construir_prompt_chat(tokenizer, record["profile_text"], treino=True, enable_thinking=enable_thinking)
    target = montar_target_treino(record["target_tags"])
    prompt_ids = tokenizer(
        prompt, add_special_tokens=False, truncation=True,
        max_length=max_input_tokens,
    )["input_ids"]
    target_ids = tokenizer(
        target + tokenizer.eos_token, add_special_tokens=False, truncation=True,
        max_length=max_target_tokens,
    )["input_ids"]
    return {
        "input_ids": prompt_ids + target_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(target_ids)),
        "labels": [-100] * len(prompt_ids) + target_ids,
        "author_id": record["author_id"],
        "input_tokens": len(prompt_ids),
        "target_tokens": len(target_ids),
        "total_tokens": len(prompt_ids) + len(target_ids),
    }


def preparar_dataset_trainer(
    records: list[dict], tokenizer: Any, *, max_input_tokens: int = 8192,
    max_target_tokens: int = 1024, enable_thinking: bool = False,
):
    from datasets import Dataset
    return Dataset.from_list([
        tokenizar_exemplo_treino(
            row, tokenizer, max_input_tokens=max_input_tokens,
            max_target_tokens=max_target_tokens, enable_thinking=enable_thinking,
        ) for row in records
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True, help="filtered_documents.json em JSONL")
    parser.add_argument("--qrels", type=Path, required=True, help="LExR-prof-qrels_filtrado em TSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profiles-cache", type=Path, default=None,
                        help="Cache de perfis estruturados usado no fine-tuning")
    parser.add_argument("--rebuild-profiles", action="store_true")
    parser.add_argument("--split-path", type=Path, default=None,
                        help="Split oficial compartilhado entre experimentos")
    parser.add_argument("--rebuild-split", action="store_true",
                        help="REGERA o split explicitamente. Não use em benchmark congelado.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-publicacoes", type=int, default=50)
    parser.add_argument("--expected-authors", type=int, default=1431)
    parser.add_argument("--expected-train", type=int, default=1001)
    parser.add_argument("--expected-valid", type=int, default=215)
    parser.add_argument("--expected-test", type=int, default=215)
    parser.add_argument("--forced-test-author", default="9350794843721516")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.max_publicacoes <= 0:
        parser.error("max-publicacoes deve ser positivo")
    records = preparar_base_supervisionada(
        args.qrels, args.documents, args.output_dir,
        max_publicacoes=args.max_publicacoes,
        expected_authors=args.expected_authors,
        profiles_cache=args.profiles_cache,
        rebuild_profiles=args.rebuild_profiles,
    )
    cfg = SplitConfig(
        seed=args.seed,
        expected_authors=args.expected_authors,
        expected_train=args.expected_train,
        expected_valid=args.expected_valid,
        expected_test=args.expected_test,
        forced_test_author=str(args.forced_test_author),
    )
    split_path = args.split_path or args.output_dir / "split_autores_seed42.json"
    split = criar_split_por_autor(records, split_path, cfg, rebuild=args.rebuild_split)
    salvar_particoes(records, split, args.output_dir)
    print("Conjuntos gerados por autor:", {name: len(split[name]) for name in ("train", "validation", "test")})
    print("Arquivos:", args.output_dir.resolve())
    print("IMPORTANTE: ordem cronológica não é imposta; verifique a entrada.")


if __name__ == "__main__":
    main()
