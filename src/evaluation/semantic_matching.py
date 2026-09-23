"""Correspondência semântica compartilhada entre POP, TF-IDF e demais modelos.

A avaliação usa SBERT com embeddings L2-normalizados e pareamento greedy
GLOBAL 1-para-1 (NÃO algoritmo Húngaro). O pareamento não altera o ranking
original das predições. Apenas espaços redundantes das tags são normalizados.

A saída de ``matching_greedy_1_to_1`` mantém as posições originais do ranking,
o que permite calcular nDCG, P, R e MAP depois em ``metrics.py``.

Dependências para encodificação: sentence-transformers, torch e numpy.
Não é necessário carregar o modelo SBERT ao importar este módulo.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

SBERT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
BATCH_SBERT = 256
THRESHOLD_SBERT = 0.75
LIMIAR_REJEICAO = 0.0
NO_MATCH_LABEL = "sem match válido"
RE_ESPACOS = re.compile(r"\s+")


def normalizar_tag(texto: str) -> str:
    """Remove espaços duplicados/bordas; preserva acentos, caixa e pontuação."""
    if not texto:
        return ""
    return RE_ESPACOS.sub(" ", texto).strip()


def carregar_qrels(caminho: str | Path) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, str]]]:
    """Lê TSV ``autor\t0\ttag\tpeso`` e devolve gabarito + forma original.

    Em tags repetidas após correção de espaços, guarda o maior peso e
    preserva a primeira grafia encontrada, como nos notebooks originais.
    """
    gt_norm: Dict[str, Dict[str, int]] = defaultdict(dict)
    gt_original: Dict[str, Dict[str, str]] = defaultdict(dict)

    with open(caminho, "r", encoding="utf-8") as arquivo:
        for linha in arquivo:
            partes = linha.strip().split("\t")
            if len(partes) < 4:
                continue
            autor, tag_orig = partes[0], partes[2]
            try:
                nota = int(partes[3])
            except ValueError:
                continue
            tag = normalizar_tag(tag_orig)
            if not tag:
                continue
            if tag in gt_norm[autor]:
                if nota > gt_norm[autor][tag]:
                    gt_norm[autor][tag] = nota
            else:
                gt_norm[autor][tag] = nota
                gt_original[autor][tag] = tag_orig

    return dict(gt_norm), dict(gt_original)


def construir_vocabulario(
    rankings: Mapping[str, Sequence[Tuple[str, float]]],
    gt_norm: Mapping[str, Mapping[str, int]],
    perfis_doc_semantico: Mapping[str, Mapping[str, Sequence[str]]],
    top_k_pred: int,
) -> List[str]:
    """União de tags do gabarito, top-k preditas e campos naturais de Coverage."""
    vocab: Set[str] = set()
    for dic in gt_norm.values():
        vocab.update(dic.keys())
    for ranking in rankings.values():
        for tag, _ in ranking[:top_k_pred]:
            vocab.add(tag)
    for docs in perfis_doc_semantico.values():
        for campos in docs.values():
            vocab.update(campos)
    vocab.discard("")
    return sorted(vocab)


def encodar_com_sbert(
    modelo: Any,
    tags: List[str],
    batch_size: int = BATCH_SBERT,
    device: Optional[str] = None,
    show_progress_bar: bool = True,
) -> Dict[str, np.ndarray]:
    """Codifica as strings uma vez com SBERT, normalizando cada vetor em L2.

    ``modelo`` é uma instância já carregada de ``SentenceTransformer``.
    Devolve ``{string: embedding float32 normalizado}``.
    """
    if not tags:
        return {}
    import torch
    import torch.nn.functional as F

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = modelo.encode(
        tags,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=show_progress_bar,
        device=device,
    )
    emb = F.normalize(emb, p=2, dim=1)
    emb_np = emb.cpu().numpy().astype(np.float32)
    return {tag: emb_np[i] for i, tag in enumerate(tags)}


def stack_embeddings(tags: Sequence[str], cache: Mapping[str, np.ndarray]) -> np.ndarray:
    """Empilha embeddings na ordem solicitada; ausentes recebem vetor zero."""
    if not cache:
        raise ValueError("O cache de embeddings está vazio.")
    d = next(iter(cache.values())).shape[0]
    matriz = np.zeros((len(tags), d), dtype=np.float32)
    for i, tag in enumerate(tags):
        if tag in cache:
            matriz[i] = cache[tag]
    return matriz


def construir_matriz_similaridade(
    autor: str,
    ranking: Sequence[Tuple[str, float]],
    gt_norm_autor: Mapping[str, int],
    cache_emb: Mapping[str, np.ndarray],
    top_k: int = 20,
) -> Optional[dict]:
    """Matriz cosseno (predições no ranking original × gabarito graduado).

    O gabarito é ordenado por peso decrescente e, em empate, tag alfabética.
    Como os embeddings já estão normalizados, o produto escalar = cosseno.
    """
    if not gt_norm_autor or not ranking:
        return None
    pred_tags = [tag for tag, _ in ranking[:top_k]]
    if not pred_tags:
        return None
    gold_items = sorted(gt_norm_autor.items(), key=lambda item: (-item[1], item[0]))
    gold_tags = [tag for tag, _ in gold_items]
    gold_weights = np.array([peso for _, peso in gold_items], dtype=np.int32)
    pred_emb = stack_embeddings(pred_tags, cache_emb)
    gold_emb = stack_embeddings(gold_tags, cache_emb)
    sim = (pred_emb @ gold_emb.T).astype(np.float32)
    return {
        "autor": str(autor),
        "sim": sim,
        "pred_tags": pred_tags,
        "gold_tags": gold_tags,
        "gold_weights": gold_weights,
        "pred_emb": pred_emb,
        "gold_emb": gold_emb,
    }


def matching_greedy_1_to_1(
    sim: np.ndarray,
    gold_weights: np.ndarray,
    theta: float = THRESHOLD_SBERT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy GLOBAL 1-para-1, igual nos notebooks POP e TF-IDF.

    Ordena TODOS os pares por similaridade decrescente (ordenação estável),
    seleciona apenas predita/gabarito ainda livres e exige ``sim >= theta``
    E ``sim > 0``. Mantém os vetores de retorno no ranking ORIGINAL.

    Retorna ``(matched_weights, matched_gold_idx, matched_sims)``.
    Uma predita rejeitada retém como ``matched_sims`` o seu melhor cosseno bruto
    para auditoria; seu peso continua zero e o índice de gabarito é -1.
    """
    sim = np.asarray(sim, dtype=np.float32)
    gold_weights = np.asarray(gold_weights, dtype=np.int32)
    if sim.ndim != 2:
        raise ValueError("sim deve ser uma matriz 2D (predições × gabarito).")
    k, g = sim.shape
    if gold_weights.shape != (g,):
        raise ValueError("gold_weights deve ter uma entrada para cada coluna de sim.")
    matched_weights = np.zeros(k, dtype=np.int32)
    matched_idx = np.full(k, -1, dtype=np.int32)
    matched_sims = (
        np.max(sim, axis=1).astype(np.float32)
        if g > 0 else np.zeros(k, dtype=np.float32)
    )
    if k == 0 or g == 0:
        return matched_weights, matched_idx, matched_sims

    ordem_global = np.argsort(-sim, axis=None, kind="stable")
    pred_usada = np.zeros(k, dtype=bool)
    gold_usado = np.zeros(g, dtype=bool)
    for flat_idx in ordem_global:
        i, j = np.unravel_index(int(flat_idx), sim.shape)
        score = float(sim[i, j])
        if score < theta or score <= LIMIAR_REJEICAO:
            break
        if pred_usada[i] or gold_usado[j]:
            continue
        matched_weights[i] = int(gold_weights[j])
        matched_idx[i] = int(j)
        matched_sims[i] = score
        pred_usada[i] = True
        gold_usado[j] = True
    return matched_weights, matched_idx, matched_sims


def salvar_npz(
    dados_autor: Mapping[str, Any],
    matched_idx: np.ndarray,
    matched_weights: np.ndarray,
    matched_sims: np.ndarray,
    pasta_saida: str | Path,
) -> Path:
    """Armazena as matrizes e o pareamento do autor sem exigir pickle."""
    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)
    saida = pasta_saida / f"{dados_autor['autor']}.npz"
    np.savez_compressed(
        saida,
        sim=dados_autor["sim"],
        pred_tags=np.asarray(dados_autor["pred_tags"], dtype=np.str_),
        gold_tags=np.asarray(dados_autor["gold_tags"], dtype=np.str_),
        gold_weights=dados_autor["gold_weights"],
        matched_gold_idx=matched_idx,
        matched_weights=matched_weights,
        matched_sims=matched_sims,
    )
    return saida


def salvar_csv_avaliacoes_gerais(
    dados_por_autor: Mapping[str, dict],
    matching_por_autor: Mapping[str, dict],
    gt_original: Mapping[str, Mapping[str, str]],
    caminho: str | Path,
    modelo: str,
    rank_max: int = 20,
) -> None:
    """CSV detalhado por autor/rank, para auditoria das correspondências."""
    colunas = [
        "Autor", "Rank", "Tag_Predita", "Melhor_Match_Gabarito",
        "Similaridade", "Peso", "Match_Status", "Modelo",
    ]
    with open(caminho, "w", encoding="utf-8", newline="") as arquivo:
        writer = csv.writer(arquivo, delimiter=";")
        writer.writerow(colunas)
        for autor, dados in dados_por_autor.items():
            info = matching_por_autor.get(autor, {})
            pred_tags = dados["pred_tags"]
            gold_tags = dados["gold_tags"]
            sim = dados["sim"]
            idx_array = info.get("matched_idx")
            weights = info.get("matched_weights")
            sims = info.get("matched_sims")
            gt_orig = gt_original.get(autor, {})
            for rank in range(1, rank_max + 1):
                i = rank - 1
                autor_exib = f"ID_{autor}"
                if i >= len(pred_tags):
                    writer.writerow([autor_exib, rank, "", "", "", "", "", modelo])
                    continue
                pred = pred_tags[i]
                sim_val = float(sims[i]) if sims is not None else 0.0
                idx_g = int(idx_array[i]) if idx_array is not None else -1
                peso = int(weights[i]) if weights is not None else 0
                if idx_g >= 0:
                    gold = gold_tags[idx_g]
                    match_disp = gt_orig.get(gold, gold)
                    status = "match válido"
                else:
                    if sim.shape[1] > 0:
                        best_j = int(np.argmax(sim[i]))
                        gold = gold_tags[best_j]
                        match_disp = f"{gt_orig.get(gold, gold)} (rej)"
                    else:
                        match_disp = NO_MATCH_LABEL
                    status = "sem match válido"
                writer.writerow([
                    autor_exib, rank, pred, match_disp,
                    f"{sim_val:.3f}", peso, status, modelo,
                ])
