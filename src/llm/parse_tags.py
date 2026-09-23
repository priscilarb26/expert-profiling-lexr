"""Extração de tags e ranking posicional."""

from __future__ import annotations

import json
import re


RE_ESPACOS = re.compile(r"\s+")


def normalizar_tag(texto: str) -> str:
    """Apenas ajusta espaços; preserva caixa, acentos e pontuação."""
    if not texto:
        return ""
    return RE_ESPACOS.sub(" ", texto).strip()


def parsear_tags_da_resposta(resposta: str, *, n_tags_pedir: int = 30) -> list[str]:
    """Aceita JSON, bloco ```json, array parcial ou lista solta.

    Replica a prioridade e os fallbacks do notebook. O corte em ``n_tags_pedir``
    é aplicado SOMENTE ao fallback de texto livre, tal como no original.
    """
    txt = re.sub(r"^```(?:json)?\s*", "", resposta.strip())
    txt = re.sub(r"\s*```$", "", txt)

    # JSON direto: {"tags": [...]} ou uma lista JSON.
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict) and "tags" in obj and isinstance(obj["tags"], list):
            return [str(t).strip() for t in obj["tags"] if str(t).strip()]
        if isinstance(obj, list):
            return [str(t).strip() for t in obj if str(t).strip()]
    except json.JSONDecodeError:
        pass

    # Recuperação de tags completas dentro de JSON parcialmente truncado.
    m = re.search(r'"tags"\s*:\s*\[(.*)', txt, flags=re.DOTALL)
    if m:
        tags = re.findall(r'"([^"]+)"', m.group(1))
        if tags:
            return [t.strip() for t in tags if t.strip()]

    # Fallback do notebook para texto separado por vírgula, ; ou nova linha.
    candidatas = re.split(r"[,\n;]+", txt)
    tags = [c.strip(' "\'\t-•').strip() for c in candidatas]
    tags = [t for t in tags if t and len(t) < 80]
    return tags[:n_tags_pedir] if tags else []


def construir_ranking_qwen(tags_llm: list[str]) -> list[tuple[str, int]]:
    """Remove duplicatas e atribui escore decrescente pela posição original.

    Exemplo: ["A", "A", "B"] => [("A", 3), ("B", 1)].
    """
    ranking: list[tuple[str, int]] = []
    vistos: set[str] = set()
    n = len(tags_llm)
    for pos, tag in enumerate(tags_llm):
        tag_norm = normalizar_tag(tag)
        if not tag_norm or tag_norm in vistos:
            continue
        vistos.add(tag_norm)
        score = n - pos
        ranking.append((tag_norm, score))
    return ranking
