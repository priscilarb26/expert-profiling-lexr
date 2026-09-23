"""Constrói os perfis estruturados usados como entrada pelos modelos Qwen.

A preparação NÃO depende de uma família, tokenizer, GPU ou configuração de geração do modelo;
portanto, o mesmo arquivo de saída pode ser reutilizado nas diferentes variantes Qwen.

Entradas
--------
- ``filtered_documents.json`` no formato JSONL, com uma publicação por linha,
  incluindo ``authors``, ``title``, ``keywords`` e ``abstract``.
- ``LExR-prof-qrels_filtrado`` (TSV), cuja primeira coluna contém o ID do autor.

Saída
-----
``perfis_estruturados_qwen.json`` no mesmo formato do notebook original::

    {
      "ID_DO_AUTOR": [
        {"titulo": "...", "keywords": "palavra 1, palavra 2", "abstract": "..."},
        ...
      ]
    }

Cada publicação é associada a todos os coautores presentes no conjunto de
qrels. São mantidas publicações com pelo menos um dos campos ``titulo`` ou
``abstract`` preenchido. Uma limpeza leve remove HTML, caracteres considerados
ruído e espaços redundantes sem aplicar lowercase nem remover acentos.

Uso :

    python src/preprocessing/build_profiles_qwen.py \\
        --documents /caminho/filtered_documents.json \\
        --qrels /caminho/LExR-prof-qrels_filtrado \\
        --output /caminho/perfis_estruturados_qwen.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  
    tqdm = lambda iterable, **kwargs: iterable

# Expressões regulares preservadas do notebook original.
RE_HTML = re.compile(r"<[^>]+>")
RE_ENTIDADE_HTML = re.compile(r"&[a-z]+;|&#\d+;")
RE_CARACTERES_RUIM = re.compile(
    r"[^\w\s.,;:!?()\-'\"áéíóúâêîôûãõàèìòùçÁÉÍÓÚÂÊÎÔÛÃÕÀÈÌÒÙÇñÑüÜ]",
    flags=re.UNICODE,
)
RE_ESPACOS = re.compile(r"\s+")


def limpar_texto_basico(texto: str) -> str:
    """Limpeza leve, preservando caixa, acentuação e pontuação básica."""
    if not texto:
        return ""
    texto = RE_HTML.sub(" ", texto)
    texto = RE_ENTIDADE_HTML.sub(" ", texto)
    texto = RE_CARACTERES_RUIM.sub(" ", texto)
    return RE_ESPACOS.sub(" ", texto).strip()


def carregar_autores_qrels_set(caminho: str | Path) -> set[str]:
    """Retorna os identificadores da primeira coluna dos qrels filtrados."""
    autores: set[str] = set()
    with Path(caminho).open("r", encoding="utf-8") as arquivo:
        for linha in arquivo:
            partes = linha.strip().split("\t")
            if partes and partes[0]:
                autores.add(partes[0])
    if not autores:
        raise ValueError(f"Nenhum autor foi encontrado em {caminho}.")
    return autores


def estruturar_publicacao(doc: dict[str, Any]) -> dict[str, str]:
    """Transforma título, keywords e resumo no formato usado no prompt Qwen.

    As keywords são agrupadas numa única string
    separada por vírgula; não há deduplicação nem alteração de idioma.
    """
    titulo = limpar_texto_basico(doc.get("title") or "")
    abstract = limpar_texto_basico(doc.get("abstract") or "")

    keywords_lista = doc.get("keywords") or []
    keywords_str = ", ".join(
        limpar_texto_basico(k)
        for k in keywords_lista
        if isinstance(k, str) and k.strip()
    )
    return {"titulo": titulo, "keywords": keywords_str, "abstract": abstract}


def construir_perfis_estruturados(
    caminho_jsonl: str | Path,
    autores_alvo: set[str],
) -> dict[str, list[dict[str, str]]]:
    """Lê JSONL e gera ``{id_autor: [publicacao_estruturada, ...]}``.

    A ordem das publicações é a do arquivo de entrada. 
    Somente autores dos qrels são incluídos, e publicações sem título E sem
    resumo são descartadas, ainda que possuam keywords.
    """
    acumulador: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    linhas_lidas = 0
    linhas_invalidas = 0
    publicacoes_selecionadas = 0

    print(f">> Lendo {caminho_jsonl} e criando perfis estruturados Qwen...")
    with Path(caminho_jsonl).open("r", encoding="utf-8") as arquivo:
        for linha in tqdm(arquivo, desc="publicações lidas"):
            linhas_lidas += 1
            linha = linha.strip()
            if not linha:
                continue
            try:
                doc = json.loads(linha)
            except json.JSONDecodeError:
                linhas_invalidas += 1
                continue

            if not isinstance(doc, dict):
                continue
            autores = doc.get("authors") or []
            if not isinstance(autores, list):
                continue
            if not any(a in autores_alvo for a in autores):
                continue

            publicacao = estruturar_publicacao(doc)
            if not publicacao["titulo"] and not publicacao["abstract"]:
                continue

            for autor in autores:
                if autor in autores_alvo:
                    acumulador[autor].append(publicacao)
            publicacoes_selecionadas += 1

    perfis = dict(acumulador)
    print(f"   Linhas lidas              : {linhas_lidas:,}")
    print(f"   Linhas JSON inválidas     : {linhas_invalidas:,}")
    print(f"   Publicações selecionadas  : {publicacoes_selecionadas:,}")
    print(f"   Autores com publicações   : {len(perfis):,}")
    print(f"   Entradas autor-publicação : {sum(map(len, perfis.values())):,}")
    return perfis


def salvar_perfis(
    perfis: dict[str, list[dict[str, str]]],
    caminho_saida: str | Path,
) -> None:
    """Salva os perfis em JSON UTF-8; protege o resultado de gravação parcial."""
    destino = Path(caminho_saida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_name(destino.name + ".partial")
    try:
        with temporario.open("w", encoding="utf-8") as arquivo:
            json.dump(perfis, arquivo, ensure_ascii=False)
        temporario.replace(destino)
    finally:
        temporario.unlink(missing_ok=True)
    print(f"   Perfis salvos em: {destino}")


def gerar_perfis_qwen(
    documentos: str | Path,
    qrels: str | Path,
    saida: str | Path,
) -> dict[str, list[dict[str, str]]]:
    """Executa a preparação genérica Qwen e devolve os perfis gerados."""
    documentos, qrels, saida = map(Path, (documentos, qrels, saida))
    if documentos.resolve() == saida.resolve() or qrels.resolve() == saida.resolve():
        raise ValueError("A saída não pode sobrescrever um arquivo de entrada.")
    autores = carregar_autores_qrels_set(qrels)
    print(f"   IDs presentes nos qrels: {len(autores):,}")
    perfis = construir_perfis_estruturados(documentos, autores)
    salvar_perfis(perfis, saida)
    return perfis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--documents", type=Path, required=True,
        help="Entrada filtered_documents.json, formato JSONL (1 publicação/linha).",
    )
    parser.add_argument(
        "--qrels", type=Path, required=True,
        help="Arquivo LExR-prof-qrels_filtrado (TSV).",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="JSON de saída com os perfis estruturados para os modelos Qwen.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gerar_perfis_qwen(args.documents, args.qrels, args.output)


if __name__ == "__main__":
    main()
