"""Baseline POP para o projeto Expert Profiling (LExR).

Este módulo implementa exclusivamente a geração e o ranqueamento de tags
pelo método POP, a partir dos n-gramas previamente produzidos em
``src/preprocessing/build_profiles_baseline.py``.

A frequência de cada n-grama é seu escore. O ranking é ordenado por frequência
absoluta decrescente, com desempate alfabético crescente.

Não realiza pré-processamento, SBERT, pareamento, cálculo de métricas ou
gerenciamento de arquivos específicos. Essas operações
pertencem a outras etapas do projeto.

Entrada JSON (perfis_autores.json):
    {"id_autor": ["tag_a", "tag_b", "tag_a", ...], ...}

Saída JSON (tags_brutas_pop.json):
    {"id_autor": [{"tag": "tag_a", "freq": 2},
                  {"tag": "tag_b", "freq": 1}], ...}

Exemplo de execução na raiz do repositório:
    python src/baselines/pop.py \\
        --profiles data/processed/perfis_autores.json \\
        --output results/full_dataset/baselines/tags_brutas_pop.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


# A função é a mesma utilizada na etapa POP do notebook original.
def calcular_ranking_pop(ngrams: List[str]) -> List[Tuple[str, int]]:
    """Ordena n-gramas por frequência decrescente e desempate alfabético."""
    contagem = Counter(ngrams)
    ranking = sorted(contagem.items(), key=lambda x: (-x[1], x[0]))
    return ranking


def gerar_rankings_pop(
    perfis: Dict[str, List[str]],
) -> Dict[str, List[Tuple[str, int]]]:
    """Aplica POP a cada autor, sem alterar ou limitar as tags originais."""
    rankings: Dict[str, List[Tuple[str, int]]] = {}
    for autor, ngrams in perfis.items():
        rankings[autor] = calcular_ranking_pop(ngrams)
    return rankings


def preparar_tags_brutas(
    rankings: Dict[str, List[Tuple[str, int]]],
) -> Dict[str, List[dict]]:
    """Converte rankings para o mesmo formato de tags brutas do notebook."""
    return {
        autor: [{"tag": tag, "freq": int(freq)} for tag, freq in ranking]
        for autor, ranking in rankings.items()
    }


def carregar_perfis(caminho: str | Path) -> Dict[str, List[str]]:
    """Lê o arquivo de perfis pré-processados, preservando repetições."""
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
    """Salva rankings no esquema original usado na avaliação POP."""
    destino = Path(caminho)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8") as arquivo:
        json.dump(tags_brutas, arquivo, ensure_ascii=False)


def executar_pop(
    arquivo_perfis: str | Path,
    arquivo_saida: str | Path,
) -> Dict[str, List[dict]]:
    """Executa o baseline POP e salva o ranking completo de todos os autores."""
    perfis = carregar_perfis(arquivo_perfis)
    rankings = gerar_rankings_pop(perfis)
    tags_brutas = preparar_tags_brutas(rankings)
    salvar_tags_brutas(tags_brutas, arquivo_saida)
    return tags_brutas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera o ranking POP por autor a partir de perfis de n-gramas."
    )
    parser.add_argument(
        "--profiles", type=Path, required=True,
        help="Arquivo JSON de perfis pré-processados (perfis_autores.json).",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Destino do JSON com rankings (tags_brutas_pop.json).",
    )
    args = parser.parse_args()

    resultados = executar_pop(args.profiles, args.output)
    print(f"POP concluído para {len(resultados)} autores.")
    print(f"Ranking bruto salvo em: {args.output}")


if __name__ == "__main__":
    main()
