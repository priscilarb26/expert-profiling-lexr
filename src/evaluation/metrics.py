"""Métricas de avaliação extraídas dos notebooks POP e TF-IDF.

Mesmas definições e cortes utilizados nos dois notebooks:
- nDCG@10/@20: relevância graduada do gabarito (1, 2, 3);
- P@5/@10/@20, R@5/@10/@20 e MAP@10/@20: relevante se peso >= 2;
- Coverage@10/@20: match exato em n-gramas da publicação, com fallback
  semântico SBERT em título, keywords individuais e abstract;
- Diversity@10/@20: 1 - similaridade média fora da diagonal;
- Match_Valido@10/@20: quantidade de matches válidos (peso >= 1).

O pareamento é feito em ``semantic_matching.py`` e não altera o ranking.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

try:  
    from .semantic_matching import stack_embeddings
except ImportError:  
    from semantic_matching import stack_embeddings

THRESHOLD_RELEVANCIA = 2
THRESHOLD_COBERTURA = 0.75
KS_NDCG = (10, 20)
KS_PRECISAO = (5, 10, 20)
KS_RECALL = (5, 10, 20)
K_MAP = 10
K_COVERAGE = 10
K_DIVERSITY = 10
TOP_K_ALVO_METRICAS = max(max(KS_NDCG), max(KS_PRECISAO), max(KS_RECALL),
                          K_MAP, K_COVERAGE, K_DIVERSITY)

RE_ESPACOS = re.compile(r"\s+")
RE_HTML_COV = re.compile(r"<[^>]+>")
RE_ENTIDADE_HTML_COV = re.compile(r"&[a-z]+;|&#\d+;")
RE_CARACTERES_RUIM_COV = re.compile(
    r"[^\w\s.,;:!?()\-'\"áéíóúâêîôûãõàèìòùçÁÉÍÓÚÂÊÎÔÛÃÕÀÈÌÒÙÇñÑüÜ]",
    flags=re.UNICODE,
)


def limpar_texto_basico_coverage(texto: str) -> str:
    """Limpeza leve usada SOMENTE nos campos naturais do Coverage semântico."""
    if not texto:
        return ""
    texto = str(texto)
    texto = RE_HTML_COV.sub(" ", texto)
    texto = RE_ENTIDADE_HTML_COV.sub(" ", texto)
    texto = RE_CARACTERES_RUIM_COV.sub(" ", texto)
    texto = RE_ESPACOS.sub(" ", texto).strip()
    return texto


def carregar_perfis_por_documento(caminho: str | Path) -> Dict[str, Dict[str, List[str]]]:
    """Lê ``{autor: {doc_id: [n-gramas]}}`` para match exato de Coverage."""
    if not os.path.exists(caminho):
        print(f"[Aviso] {caminho} não encontrado — Coverage@k ficará indisponível.")
        return {}
    with open(caminho, "r", encoding="utf-8") as arquivo:
        return json.load(arquivo)


def carregar_campos_semanticos_por_documento(
    caminho_jsonl: str | Path,
    autores_alvo: Sequence[str] | set[str],
) -> Dict[str, Dict[str, List[str]]]:
    """Lê JSONL de publicações e devolve campos naturais por autor/doc_id.

    Preserva a leitura linha a linha de ``filtered_documents.json`` original.
    Esse arquivo é JSONL no notebook: cada linha representa um documento JSON.
    """
    saida: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    if not os.path.exists(caminho_jsonl):
        print(f"[Aviso] {caminho_jsonl} não encontrado — Coverage semântico indisponível.")
        return {}
    autores_alvo = set(autores_alvo)
    with open(caminho_jsonl, "r", encoding="utf-8") as arquivo:
        for linha in arquivo:
            linha = linha.strip()
            if not linha:
                continue
            try:
                doc = json.loads(linha)
            except json.JSONDecodeError:
                continue
            doc_id = str(doc.get("id") or "")
            if not doc_id:
                continue
            autores_doc = [a for a in (doc.get("authors") or []) if a in autores_alvo]
            if not autores_doc:
                continue
            campos: List[str] = []
            titulo = limpar_texto_basico_coverage(doc.get("title") or "")
            if titulo:
                campos.append(titulo)
            for kw in doc.get("keywords") or []:
                if not isinstance(kw, str):
                    continue
                kw_limpa = limpar_texto_basico_coverage(kw)
                if kw_limpa:
                    campos.append(kw_limpa)
            abstract = limpar_texto_basico_coverage(doc.get("abstract") or "")
            if abstract:
                campos.append(abstract)
            if not campos:
                continue
            for autor in autores_doc:
                saida[autor][doc_id] = campos
    return {autor: dict(docs) for autor, docs in saida.items()}


def dcg_at_k(rels: np.ndarray, k: int) -> float:
    """DCG com ganho exponencial e desconto logarítmico."""
    soma = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        if rel > 0:
            soma += (2 ** int(rel) - 1) / math.log2(i + 1)
    return soma


def ndcg_at_k(matched_weights: np.ndarray, gold_weights: np.ndarray, k: int) -> float:
    idcg = dcg_at_k(np.sort(gold_weights)[::-1], k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(matched_weights, k) / idcg


def precision_at_k(matched_weights: np.ndarray, k: int) -> float:
    if k <= 0:
        return 0.0
    return float(np.sum(matched_weights[:k] >= THRESHOLD_RELEVANCIA)) / k


def recall_at_k(matched_weights: np.ndarray, gold_weights: np.ndarray, k: int) -> float:
    total_rel = int(np.sum(gold_weights >= THRESHOLD_RELEVANCIA))
    if total_rel == 0:
        return 0.0
    return float(np.sum(matched_weights[:k] >= THRESHOLD_RELEVANCIA)) / total_rel


def average_precision_at_k(
    matched_weights: np.ndarray,
    gold_weights: np.ndarray,
    k: int = K_MAP,
) -> float:
    total_rel_gt = int(np.sum(gold_weights >= THRESHOLD_RELEVANCIA))
    if total_rel_gt == 0:
        return 0.0
    soma_precisao = 0.0
    n_hits = 0
    for i in range(min(k, len(matched_weights))):
        if matched_weights[i] >= THRESHOLD_RELEVANCIA:
            n_hits += 1
            soma_precisao += n_hits / (i + 1)
    return soma_precisao / total_rel_gt


def coverage_semantica_at_k(
    top_k_pred_tags: Sequence[str],
    top_k_pred_emb: np.ndarray,
    docs_ngrams_autor: Mapping[str, Sequence[str]],
    docs_campos_autor: Mapping[str, Sequence[str]],
    cache_emb: Mapping[str, np.ndarray],
    theta_cov: float = THRESHOLD_COBERTURA,
) -> float:
    """Fração de docs tocados por pelo menos uma das top-k tags.

    1) match literal tag × n-gramas por publicação;
    2) se (1) falhar, fallback SBERT tag × título/keywords/abstract.
    Denominador = número de publicações com n-gramas, mesmo quando vazios.
    """
    if not docs_ngrams_autor:
        return 0.0
    total_docs = len(docs_ngrams_autor)
    set_pred = set(top_k_pred_tags)
    cobertos = 0
    for doc_id, ngrams in docs_ngrams_autor.items():
        ngrams_unicos = set(ngrams)
        if ngrams_unicos and (set_pred & ngrams_unicos):
            cobertos += 1
            continue
        campos_unicos = set(docs_campos_autor.get(str(doc_id), []))
        if not campos_unicos:
            continue
        campos_lista = list(campos_unicos)
        campos_emb = stack_embeddings(campos_lista, cache_emb)
        if campos_emb.shape[0] == 0:
            continue
        sim_block = top_k_pred_emb @ campos_emb.T
        if (sim_block >= theta_cov).any():
            cobertos += 1
    return cobertos / total_docs


def diversity_intra_list_at_k(pred_emb_topk: np.ndarray) -> float:
    """1 - média das similaridades de todos os pares distintos na lista."""
    k = pred_emb_topk.shape[0]
    if k < 2:
        return 0.0
    sim_mat = pred_emb_topk @ pred_emb_topk.T
    off_diag_sum = float(sim_mat.sum() - np.trace(sim_mat))
    return 1.0 - off_diag_sum / (k * (k - 1))


def avaliar_autor(
    dados_autor: dict,
    matched_weights: np.ndarray,
    docs_ngrams_autor: Mapping[str, Sequence[str]],
    docs_campos_autor: Mapping[str, Sequence[str]],
    cache_emb: Mapping[str, np.ndarray],
    theta_cov: float = THRESHOLD_COBERTURA,
) -> Dict[str, float]:
    """Calcula todas as métricas, preservando nomes e cortes dos notebooks."""
    gold_weights = dados_autor["gold_weights"]
    pred_tags = dados_autor["pred_tags"]
    pred_emb = dados_autor["pred_emb"]
    metricas: Dict[str, float] = {}

    for k in KS_NDCG:
        metricas[f"nDCG@{k}"] = ndcg_at_k(matched_weights, gold_weights, k)
    for k in KS_PRECISAO:
        metricas[f"P@{k}"] = precision_at_k(matched_weights, k)
    for k in KS_RECALL:
        metricas[f"R@{k}"] = recall_at_k(matched_weights, gold_weights, k)
    metricas[f"MAP@{K_MAP}"] = average_precision_at_k(matched_weights, gold_weights, k=K_MAP)
    metricas["MAP@20"] = average_precision_at_k(matched_weights, gold_weights, k=20)

    metricas[f"C@{K_COVERAGE}"] = coverage_semantica_at_k(
        pred_tags[:K_COVERAGE], pred_emb[:K_COVERAGE], docs_ngrams_autor,
        docs_campos_autor, cache_emb, theta_cov=theta_cov,
    )
    metricas["C@20"] = coverage_semantica_at_k(
        pred_tags[:20], pred_emb[:20], docs_ngrams_autor,
        docs_campos_autor, cache_emb, theta_cov=theta_cov,
    )
    metricas[f"D@{K_DIVERSITY}"] = diversity_intra_list_at_k(pred_emb[:K_DIVERSITY])
    metricas["D@20"] = diversity_intra_list_at_k(pred_emb[:20])
    metricas["Match_Valido@10"] = int(np.sum(matched_weights[:10] >= 1))
    metricas["Match_Valido@20"] = int(np.sum(matched_weights[:20] >= 1))
    return metricas


def salvar_csv_metricas(
    metricas_por_autor: Mapping[str, Mapping[str, float]],
    caminho: str | Path,
    modelo: str,
) -> None:
    """CSV por autor compatível com saída dos notebooks, delimitador ``;``."""
    colunas = [
        "Autor", "nDCG@10", "nDCG@20",
        "Precision@5", "Precision@10", "Precision@20",
        "Recall@5", "Recall@10", "Recall@20",
        "MAP@10", "Coverage@10", "Diversity@10",
        "MAP@20", "Coverage@20", "Diversity@20",
        "Match_Valido@10", "Match_Valido@20", "Modelo",
    ]
    with open(caminho, "w", encoding="utf-8", newline="") as arquivo:
        writer = csv.writer(arquivo, delimiter=";")
        writer.writerow(colunas)
        for autor, m in metricas_por_autor.items():
            writer.writerow([
                f"ID_{autor}",
                f"{m['nDCG@10']:.4f}", f"{m['nDCG@20']:.4f}",
                f"{m['P@5']:.4f}", f"{m['P@10']:.4f}", f"{m['P@20']:.4f}",
                f"{m['R@5']:.4f}", f"{m['R@10']:.4f}", f"{m['R@20']:.4f}",
                f"{m['MAP@10']:.4f}", f"{m['C@10']:.4f}", f"{m['D@10']:.4f}",
                f"{m['MAP@20']:.4f}", f"{m['C@20']:.4f}", f"{m['D@20']:.4f}",
                int(m["Match_Valido@10"]), int(m["Match_Valido@20"]), modelo,
            ])
