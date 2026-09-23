"""Gera evidências textuais por documento para o Coverage dos modelos Qwen.

Entradas
--------
* ``filtered_documents.json``: arquivo JSONL (um documento JSON por linha),
  com os campos ``id``, ``authors``, ``title``, ``keywords`` e ``abstract``.
* ``LExR-prof-qrels_filtrado``: arquivo tabulado, usado APENAS para filtrar
  os IDs dos autores considerados na avaliação.

Saída
-----
``perfis_documento_qwen.json`` com a estrutura::

    {
        "id_autor": {
            "id_documento": ["título", "palavra-chave 1", ..., "resumo"]
        }
    }

O texto passa por limpeza BÁSICA: remoção de HTML, caracteres considerados
ruído e espaços redundantes. Maiúsculas, acentos, pontuação básica e a ordem
original das partes são preservados. Não ocorre truncamento de publicações,
geração de tags por LLM nem cálculo da métrica Coverage neste script.

Uso
---
    python src/preprocessing/build_document_profiles_qwen.py \\
        --documents /caminho/filtered_documents.json \\
        --qrels /caminho/LExR-prof-qrels_filtrado \\
        --output /caminho/perfis_documento_qwen.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

from tqdm import tqdm


# Mesmas expressões regulares do notebook original.
RE_HTML = re.compile(r"<[^>]+>")
RE_ENTIDADE_HTML = re.compile(r"&[a-z]+;|&#\d+;")
RE_CARACTERES_RUIM = re.compile(
    r"[^\w\s.,;:!?()\-'\"áéíóúâêîôûãõàèìòùçÁÉÍÓÚÂÊÎÔÛÃÕÀÈÌÒÙÇñÑüÜ]",
    flags=re.UNICODE,
)
RE_ESPACOS = re.compile(r"\s+")


def limpar_texto_basico(texto: str) -> str:
    """Aplica limpeza leve, sem converter para minúsculas ou remover acentos."""
    if not texto:
        return ""
    texto = RE_HTML.sub(" ", texto)
    texto = RE_ENTIDADE_HTML.sub(" ", texto)
    texto = RE_CARACTERES_RUIM.sub(" ", texto)
    texto = RE_ESPACOS.sub(" ", texto).strip()
    return texto


def carregar_autores_qrels(caminho: str) -> Set[str]:
    """Lê os IDs de autores da primeira coluna do arquivo de qrels."""
    autores: Set[str] = set()
    with open(caminho, "r", encoding="utf-8") as f:
        for linha in f:
            partes = linha.strip().split("\t")
            if partes and partes[0]:
                autores.add(partes[0])
    return autores


def gerar_perfis_documento_qwen(
    caminho_jsonl: str, autores_alvo: Set[str]
) -> Dict[str, Dict[str, List[str]]]:
    """Produz ``{autor: {doc_id: [titulo, kw1, ..., abstract]}}``.

    Para documentos com coautoria, inclui o mesmo documento em cada perfil
    cujo autor também esteja presente nos qrels filtrados.
    """
    saida: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    docs_lidos = 0
    docs_aproveitados = 0
    docs_sem_id = 0
    docs_sem_autor_alvo = 0
    docs_vazios = 0

    print(f">> Lendo {caminho_jsonl} e montando pedaços por documento (limpeza básica)...")
    with open(caminho_jsonl, "r", encoding="utf-8") as f:
        for linha in tqdm(f, desc="docs"):
            docs_lidos += 1
            linha = linha.strip()
            if not linha:
                continue
            try:
                doc = json.loads(linha)
            except json.JSONDecodeError:
                continue

            doc_id = doc.get("id") or ""
            if not doc_id:
                docs_sem_id += 1
                continue

            autores = doc.get("authors") or []
            autores_alvo_doc = [a for a in autores if a in autores_alvo]
            if not autores_alvo_doc:
                docs_sem_autor_alvo += 1
                continue

            partes: List[str] = []

            titulo = limpar_texto_basico(doc.get("title") or "")
            if titulo:
                partes.append(titulo)

            for kw in doc.get("keywords") or []:
                if not isinstance(kw, str):
                    continue
                kw_limpo = limpar_texto_basico(kw)
                if kw_limpo:
                    partes.append(kw_limpo)

            abstract = limpar_texto_basico(doc.get("abstract") or "")
            if abstract:
                partes.append(abstract)

            if not partes:
                docs_vazios += 1
                continue

            for a in autores_alvo_doc:
                saida[a][doc_id] = partes
            docs_aproveitados += 1

    print(f"   {docs_lidos} linhas lidas")
    print(f"   {docs_aproveitados} docs aproveitados")
    print(f"   {docs_sem_id} docs sem id (descartados)")
    print(f"   {docs_sem_autor_alvo} docs sem autor-alvo no qrels (descartados)")
    print(f"   {docs_vazios} docs sem pedaços não-vazios (descartados)")
    return dict(saida)


def main(documents: Path, qrels: Path, output: Path) -> None:
    """Executa a preparação e salva as evidências para o cálculo de Coverage."""
    print("=" * 60)
    print(">> Gerando perfis_documento_qwen.json (Coverage do modelo Qwen)")
    print("=" * 60)

    autores_alvo = carregar_autores_qrels(str(qrels))
    print(f"   {len(autores_alvo)} autores no qrels.")

    perfis = gerar_perfis_documento_qwen(str(documents), autores_alvo)

    total_docs = sum(len(d) for d in perfis.values())
    total_pedacos = sum(
        sum(len(pedacos) for pedacos in docs.values())
        for docs in perfis.values()
    )
    media_docs = total_docs / max(1, len(perfis))
    media_pedacos = total_pedacos / max(1, total_docs)

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f">> Salvando em {output}...")
    with output.open("w", encoding="utf-8") as f:
        json.dump(perfis, f, ensure_ascii=False)

    print("=" * 60)
    print("RESUMO")
    print(f"  Autores com docs    : {len(perfis)}")
    print(f"  Total entradas      : {total_docs} (autor × doc_id)")
    print(f"  Média docs/autor    : {media_docs:.1f}")
    print(f"  Total pedaços       : {total_pedacos}")
    print(f"  Média pedaços/doc   : {media_pedacos:.1f}")
    print(f"  Arquivo             : {output}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--documents", type=Path, required=True,
        help="Caminho do filtered_documents.json (formato JSONL)",
    )
    parser.add_argument(
        "--qrels", type=Path, required=True,
        help="Caminho do LExR-prof-qrels_filtrado",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Destino do perfis_documento_qwen.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.documents, args.qrels, args.output)
