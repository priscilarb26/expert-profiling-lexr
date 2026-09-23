"""Baseline TF-IDF do projeto Expert Profiling (LExR).

Este módulo calcula exclusivamente o ranking lexical, utilizando os mesmos perfis de
n-gramas produzidos em ``src/preprocessing/build_profiles_baseline.py``.

Fórmulas originais do notebook:
    TF(t, autor)  = frequência(t, autor) / maior_frequência_no_perfil
    DF(t)         = número de autores cujo perfil contém t ao menos uma vez
    IDF(t)        = log10(N_autores / DF(t))
    TF-IDF(t)     = TF(t, autor) * IDF(t)

A ordenação é por escore decrescente, com desempate alfabético crescente.
O escore não recebe normalização min-max, tratamento de outliers nem outros
ajustes posteriores. A normalização pelo termo mais frequente é PARTE da
fórmula do baseline original; não é a normalização adicional prevista para
experimentos híbridos futuros.

Entrada (perfis_autores.json):
    {"id_autor": ["tag_a", "tag_b", "tag_a", ...], ...}

Saída (tags_brutas_tfidf.json):
    {"id_autor": [{"tag": "tag_a", "score_tfidf": 0.123}, ...], ...}

Exemplo a partir da raiz do repositório:
    python src/baselines/tfidf.py \\
        --profiles data/processed/perfis_autores.json \\
        --output results/full_dataset/baselines/tags_brutas_tfidf.json

Não realiza pré-processamento, SBERT, pareamento, cálculo de métricas ou
gerenciamento de arquivos específicos. Essas operações
pertencem a outras etapas do projeto.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def calcular_df_global(
    perfis: Dict[str, List[str]],
) -> Tuple[Dict[str, int], int]:
    """Conta em quantos perfis distintos cada n-grama aparece (não repetições).

    N é o número total de perfis, inclusive aqueles sem n-gramas, como no
    notebook original. Isso deve representar a mesma coleção experimental
    usada na geração dos rankings.
    """
    df: Dict[str, int] = defaultdict(int)
    for ngrams in perfis.values():
        for ngram in set(ngrams):
            df[ngram] += 1
    return dict(df), len(perfis)


def calcular_idf(df: Dict[str, int], N: int) -> Dict[str, float]:
    """Calcula IDF(t) = log10(N / DF(t)), sem suavização."""
    if df and N <= 0:
        raise ValueError("O número de autores deve ser positivo quando há termos.")
    if any(not (1 <= frequencia <= N) for frequencia in df.values()):
        raise ValueError("Cada DF deve estar entre 1 e o total de autores.")
    return {ngram: math.log10(N / frequencia) for ngram, frequencia in df.items()}


def calcular_ranking_tfidf(
    ngrams_do_autor: List[str],
    idf: Dict[str, float],
) -> List[Tuple[str, float]]:
    """Repete o cálculo e a ordenação do ranking de um autor no notebook.

    TF é a frequência absoluta dividida pela maior frequência do próprio
    autor. Em caso de empate no escore, usa ordem alfabética crescente.
    """
    tf = Counter(ngrams_do_autor)
    if not tf:
        return []

    max_freq = max(tf.values())
    scores = [
        (ngram, (frequencia / max_freq) * idf.get(ngram, 0.0))
        for ngram, frequencia in tf.items()
    ]
    scores.sort(key=lambda item: (-item[1], item[0]))
    return scores


def gerar_rankings_tfidf(
    perfis: Dict[str, List[str]],
) -> Dict[str, List[Tuple[str, float]]]:
    """Calcula um único IDF global e ranqueia os termos de cada autor."""
    df_global, total_autores = calcular_df_global(perfis)
    idf = calcular_idf(df_global, total_autores)
    return {
        autor: calcular_ranking_tfidf(ngrams, idf)
        for autor, ngrams in perfis.items()
    }


def preparar_tags_brutas(
    rankings: Dict[str, List[Tuple[str, float]]],
) -> Dict[str, List[dict]]:
    """Usa o mesmo esquema `tag` + `score_tfidf` do notebook original."""
    return {
        autor: [
            {"tag": tag, "score_tfidf": float(score)}
            for tag, score in ranking
        ]
        for autor, ranking in rankings.items()
    }


def carregar_perfis(caminho: str | Path) -> Dict[str, List[str]]:
    """Lê os perfis pré-processados, mantendo as repetições necessárias ao TF."""
    with Path(caminho).open("r", encoding="utf-8") as arquivo:
        perfis = json.load(arquivo)
    if not isinstance(perfis, dict):
        raise ValueError("Esperava um JSON no formato {id_autor: [n-gramas]}.")
    for autor, ngrams in perfis.items():
        if not isinstance(autor, str) or not isinstance(ngrams, list):
            raise ValueError(f"Perfil inválido do autor {autor!r}.")
        if not all(isinstance(tag, str) for tag in ngrams):
            raise ValueError(f"As tags do autor {autor!r} devem ser strings.")
    return perfis


def salvar_tags_brutas(tags_brutas: Dict[str, List[dict]], caminho: str | Path) -> None:
    """Grava o ranking completo no formato do `tags_brutas_tfidf.json`."""
    destino = Path(caminho)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8") as arquivo:
        json.dump(tags_brutas, arquivo, ensure_ascii=False)


def executar_tfidf(
    arquivo_perfis: str | Path,
    arquivo_saida: str | Path,
) -> Dict[str, List[dict]]:
    """Carrega perfis, calcula TF-IDF e salva todos os rankings por autor."""
    perfis = carregar_perfis(arquivo_perfis)
    rankings = gerar_rankings_tfidf(perfis)
    tags_brutas = preparar_tags_brutas(rankings)
    salvar_tags_brutas(tags_brutas, arquivo_saida)
    return tags_brutas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera rankings TF-IDF por autor a partir de perfis de n-gramas."
    )
    parser.add_argument(
        "--profiles", type=Path, required=True,
        help="Arquivo JSON de perfis pré-processados (perfis_autores.json).",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Destino do JSON com rankings (tags_brutas_tfidf.json).",
    )
    args = parser.parse_args()

    resultados = executar_tfidf(args.profiles, args.output)
    print(f"TF-IDF concluído para {len(resultados)} autores.")
    print(f"Ranking bruto salvo em: {args.output}")


if __name__ == "__main__":
    main()
