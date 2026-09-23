"""Construção dos perfis lexicais dos pesquisadores para POP e TF-IDF.

Preparar as candidatas lexicais dos baselines POP e TF-IDF usando publicações do LExR.

Entrada
-------
* filtered_documents.json: arquivo JSONL (um objeto JSON por linha), com
  ``title``, ``keywords`` (lista), ``abstract``, ``authors`` (lista) e ``id``
  (necessário para a saída por documento).
* LExR-prof-qrels_filtrado: arquivo TSV cujo primeiro campo é o ID do autor.

Saídas
------
* perfis_autores.json: {autor_id: [ngrama, ...]}. As ocorrências repetidas
  são preservadas para permitir a contagem de frequência pelos baselines.
* perfis_por_documento.json: {autor_id: {doc_id: [ngrama, ...]}}, destinado
  ao cálculo de Coverage@k.

O processamento mantém as escolhas implementadas no notebook:
    - concatenação na ordem título, palavras-chave e resumo;
    - segmentação antes da limpeza;
    - detecção de idioma por sentença; 
    - limpeza lexical; 
    - spaCy PT/EN em lote;
    - n-gramas de 1 a 3 termos sem cruzar sentenças;
    - filtro morfossintático;
    - processamento paralelo opcional e semente 42 no langdetect.

Instalação:
    pip install spacy langdetect tqdm
    python -m spacy download pt_core_news_sm
    python -m spacy download en_core_web_sm

Uso
---
    python src/preprocessing/build_profiles_baseline.py \\
        --documents /caminho/filtered_documents.json \\
        --qrels /caminho/LExR-prof-qrels_filtrado \\
        --output-authors /caminho/perfis_autores.json \\
        --output-documents /caminho/perfis_por_documento.json \\
                
ATENÇÃO: as versões das dependências devem ser compatíveis com as registradas
no arquivo ``reprodutibilidade.md`` do projeto. 
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

import spacy
from langdetect import DetectorFactory, LangDetectException, detect
from tqdm import tqdm


# Parâmetros preservados do notebook de origem.
DetectorFactory.seed = 42
NGRAM_MIN = 1
NGRAM_MAX = 3
TAMANHO_MIN_TOKEN = 2
BATCH_SPACY = 128

_NLP_PT = None
_NLP_EN = None

RE_HIFEN_UNDER = re.compile(r"[-_]+")
RE_PONTUACAO = re.compile(r"[^\w\s]+", flags=re.UNICODE)
RE_DIGITOS = re.compile(r"\d+")
RE_ESPACOS = re.compile(r"\s+")
RE_ROMANO = re.compile(
    r"\b(?=[mcdlxvi])m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})\b",
    re.IGNORECASE,
)

POS_FUNCIONAIS = {
    "ADP", "CCONJ", "SCONJ", "DET", "PRON", "AUX", "PART", "PUNCT", "SYM", "X",
}
POS_BORDA_PROIBIDA = {
    "ADP", "CCONJ", "SCONJ", "DET", "PRON", "PART", "PUNCT", "SYM", "NUM", "AUX",
}


def carregar_modelos():
    """Carrega os modelos PT/EN mantendo os componentes necessários ao POS."""
    nlp_pt = spacy.load("pt_core_news_sm", disable=["parser", "ner", "lemmatizer"])
    nlp_en = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    nlp_pt.max_length = 2_000_000
    nlp_en.max_length = 2_000_000
    return nlp_pt, nlp_en


def inicializar_worker(batch_spacy: int = BATCH_SPACY) -> None:
    """Inicializa modelos e tamanho de lote uma única vez por processo."""
    global _NLP_PT, _NLP_EN, BATCH_SPACY
    BATCH_SPACY = batch_spacy
    _NLP_PT, _NLP_EN = carregar_modelos()


def montar_string_publicacao(doc: dict) -> str:
    """Concatena título, palavras-chave e resumo, nessa ordem."""
    title = doc.get("title") or ""
    abstract = doc.get("abstract") or ""
    keywords = doc.get("keywords") or []
    keywords_str = ", ".join(
        k for k in keywords if isinstance(k, str) and k.strip()
    )
    partes = [p for p in (title, keywords_str, abstract) if p]
    return ", ".join(partes)


def construir_perfis_brutos(caminho_jsonl: str | Path, autores_alvo: set) -> Dict[str, str]:
    """Concatena as publicações de cada autor presente no qrels filtrado."""
    acumulador: Dict[str, list] = defaultdict(list)
    print(">> Lendo publicações e concatenando por autor...")
    with Path(caminho_jsonl).open("r", encoding="utf-8") as arquivo:
        for linha in tqdm(arquivo, desc="docs lidos"):
            linha = linha.strip()
            if not linha:
                continue
            try:
                doc = json.loads(linha)
            except json.JSONDecodeError:
                continue  
            autores = doc.get("authors") or []
            if not autores or not any(a in autores_alvo for a in autores):
                continue
            texto_pub = montar_string_publicacao(doc)
            if not texto_pub:
                continue
            for autor in autores:
                if autor in autores_alvo:
                    acumulador[autor].append(texto_pub)

    return {autor: ", ".join(textos) for autor, textos in acumulador.items()}


def carregar_autores_qrels(caminho_qrels: str | Path) -> set:
    """Obtém IDs únicos da primeira coluna do qrels (TSV)."""
    autores = set()
    with Path(caminho_qrels).open("r", encoding="utf-8") as arquivo:
        for linha in arquivo:
            linha = linha.strip()
            if linha:
                partes = linha.split("\t")
                if partes:
                    autores.add(partes[0])
    return autores


def remover_acentos(texto: str) -> str:
    """Remove marcas diacríticas usando normalização Unicode NFD."""
    nfd = unicodedata.normalize("NFD", texto)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def normalizar_texto(texto: str) -> str:
    """Aplica a normalização lexical exata implementada no notebook."""
    if not texto:
        return ""
    texto = texto.lower()
    texto = remover_acentos(texto)
    texto = RE_HIFEN_UNDER.sub(" ", texto)
    texto = RE_PONTUACAO.sub(" ", texto)
    texto = RE_DIGITOS.sub(" ", texto)
    texto = RE_ROMANO.sub(" ", texto)
    return RE_ESPACOS.sub(" ", texto).strip()


def detectar_idioma(texto: str, default: str = "pt") -> str:
    """Detecta PT/EN por sentença, reaproveitando o idioma anterior em falhas."""
    if not texto or len(texto.strip()) < 20:
        return default
    try:
        idioma = detect(texto)
        return idioma if idioma in ("pt", "en") else default
    except LangDetectException:
        return default


def segmentar_em_sentencas(texto: str) -> List[str]:
    """Segmenta texto bruto antes de remover pontuação e vírgulas."""
    if not texto:
        return []
    pedacos = re.split(r"[.!?;\n]+|,\s+", texto)
    return [p.strip() for p in pedacos if p and p.strip()]


def ngram_bem_formado(pos_seq: List[str]) -> bool:
    """Reproduz o filtro POS aplicado às candidatas lexicais no notebook."""
    if not pos_seq:
        return False
    if pos_seq[0] in POS_BORDA_PROIBIDA or pos_seq[-1] in POS_BORDA_PROIBIDA:
        return False
    if all(p in POS_FUNCIONAIS for p in pos_seq):
        return False
    if not any(p in {"NOUN", "PROPN", "ADJ"} for p in pos_seq):
        return False
    if len(pos_seq) == 2:
        a, b = pos_seq
        if a in {"NOUN", "PROPN"} and b == "ADP":
            return False
        if a == "ADP" and b in {"VERB", "AUX"}:
            return False
    if len(pos_seq) == 3:
        a, b, c = pos_seq
        if a in {"NOUN", "PROPN"} and b == "ADP" and c in {"VERB", "AUX"}:
            return False
        if a == "ADP":
            return False
    return True


def processar_autor(args: Tuple[str, str]) -> Tuple[str, List[str]]:
    """Extrai n-gramas filtrados de um perfil, preservando repetições e ordem."""
    codigo, texto_bruto = args
    if not texto_bruto:
        return codigo, []
    sentencas = segmentar_em_sentencas(texto_bruto)
    if not sentencas:
        return codigo, []

    sentencas_pt: List[str] = []
    sentencas_en: List[str] = []
    ordem_global: List[Tuple[int, str]] = []
    idioma_corrente = "pt"
    for sentenca in sentencas:
        idioma_corrente = detectar_idioma(sentenca, default=idioma_corrente)
        sent_norm = normalizar_texto(sentenca)
        if not sent_norm:
            continue
        if idioma_corrente == "en":
            ordem_global.append((len(sentencas_en), "en"))
            sentencas_en.append(sent_norm)
        else:
            ordem_global.append((len(sentencas_pt), "pt"))
            sentencas_pt.append(sent_norm)
    if not sentencas_pt and not sentencas_en:
        return codigo, []

    # Os modelos são inicializados no worker ou no modo sequencial.
    docs_pt = list(_NLP_PT.pipe(sentencas_pt, batch_size=BATCH_SPACY)) if sentencas_pt else []
    docs_en = list(_NLP_EN.pipe(sentencas_en, batch_size=BATCH_SPACY)) if sentencas_en else []

    candidatas: List[str] = []
    for indice, idioma in ordem_global:
        documento = docs_en[indice] if idioma == "en" else docs_pt[indice]
        tokens, pos = [], []
        for token in documento:
            if token.is_space:
                continue
            texto = token.text.strip()
            if not texto or len(texto) < TAMANHO_MIN_TOKEN:
                continue
            tokens.append(texto)
            pos.append(token.pos_)

        for tamanho in range(NGRAM_MIN, NGRAM_MAX + 1):
            for inicio in range(len(tokens) - tamanho + 1):
                pos_seq = pos[inicio:inicio + tamanho]
                if ngram_bem_formado(pos_seq):
                    candidatas.append(" ".join(tokens[inicio:inicio + tamanho]))
    return codigo, candidatas


def listar_documentos_alvo(
    caminho_jsonl: str | Path, autores_alvo: set
) -> List[Tuple[str, List[str], str]]:
    """Seleciona publicações com ID, texto e pelo menos um autor elegível."""
    docs: List[Tuple[str, List[str], str]] = []
    print(">> [Por-Doc] Lendo publicações e filtrando autores-alvo...")
    with Path(caminho_jsonl).open("r", encoding="utf-8") as arquivo:
        for linha in tqdm(arquivo, desc="docs lidos"):
            linha = linha.strip()
            if not linha:
                continue
            try:
                doc = json.loads(linha)
            except json.JSONDecodeError:
                continue
            doc_id = doc.get("id") or ""
            if not doc_id:
                continue
            autores = doc.get("authors") or []
            autores_alvo_doc = [a for a in autores if a in autores_alvo]
            if not autores_alvo_doc:
                continue
            texto_pub = montar_string_publicacao(doc)
            if texto_pub:
                docs.append((doc_id, autores_alvo_doc, texto_pub))
    return docs


def processar_documento(
    args: Tuple[str, List[str], str]
) -> Tuple[str, List[str], List[str]]:
    """Reutiliza a extração por autor em um documento isolado para Coverage@k."""
    doc_id, autores_alvo_doc, texto_bruto = args
    if not texto_bruto:
        return doc_id, autores_alvo_doc, []
    _, ngrams = processar_autor((doc_id, texto_bruto))
    return doc_id, autores_alvo_doc, ngrams


def salvar_json(conteudo: dict, destino: str | Path) -> None:
    """Persiste a saída da maneira usada no notebook, sem alterar as listas."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8") as arquivo:
        json.dump(conteudo, arquivo, ensure_ascii=False)


def gerar_perfis_autores(
    caminho_documentos: str | Path,
    autores_alvo: set,
    caminho_saida: str | Path,
    workers: int,
    batch_spacy: int = BATCH_SPACY,
) -> None:
    """Executa o pipeline P@noptic por autor e grava perfis_autores.json."""
    perfis_brutos = construir_perfis_brutos(caminho_documentos, autores_alvo)
    print(f">> Perfis brutos construídos para {len(perfis_brutos)} autores.")
    tarefas = list(perfis_brutos.items())
    saida: Dict[str, List[str]] = {}

    if workers == 1:
        inicializar_worker(batch_spacy)
        for tarefa in tqdm(tarefas, desc="autores"):
            codigo, candidatas = processar_autor(tarefa)
            saida[codigo] = candidatas
    else:
        with Pool(
            processes=workers, initializer=inicializar_worker,
            initargs=(batch_spacy,),
        ) as pool:
            for codigo, candidatas in tqdm(
                pool.imap_unordered(processar_autor, tarefas, chunksize=4),
                total=len(tarefas), desc="autores",
            ):
                saida[codigo] = candidatas

    salvar_json(saida, caminho_saida)
    total_tags = sum(len(tags) for tags in saida.values())
    media = total_tags / len(saida) if saida else 0
    print(f">> Perfis salvos: {caminho_saida}")
    print(f"   Autores: {len(saida)} | Tags: {total_tags} | Média por autor: {media:.1f}")


def gerar_perfis_documentos(
    caminho_documentos: str | Path,
    autores_alvo: set,
    caminho_saida: str | Path,
    workers: int,
    batch_spacy: int = BATCH_SPACY,
) -> None:
    """Extrai candidatas por publicação para cálculo posterior de Coverage@k."""
    docs = listar_documentos_alvo(caminho_documentos, autores_alvo)
    print(f">> {len(docs)} documentos a processar.")
    saida_doc: Dict[str, Dict[str, List[str]]] = defaultdict(dict)

    if workers == 1:
        global _NLP_PT, _NLP_EN
        if _NLP_PT is None or _NLP_EN is None:
            inicializar_worker(batch_spacy)
        for tarefa in tqdm(docs, desc="docs"):
            doc_id, autores_alvo_doc, ngrams = processar_documento(tarefa)
            for autor in autores_alvo_doc:
                saida_doc[autor][doc_id] = ngrams
    else:
        with Pool(
            processes=workers, initializer=inicializar_worker,
            initargs=(batch_spacy,),
        ) as pool:
            for doc_id, autores_alvo_doc, ngrams in tqdm(
                pool.imap_unordered(processar_documento, docs, chunksize=4),
                total=len(docs), desc="docs",
            ):
                for autor in autores_alvo_doc:
                    saida_doc[autor][doc_id] = ngrams

    salvar_json(saida_doc, caminho_saida)
    total_docs = sum(len(docs_por_autor) for docs_por_autor in saida_doc.values())
    media = total_docs / max(1, len(saida_doc))
    print(f">> Perfis por documento salvos: {caminho_saida}")
    print(f"   Autores: {len(saida_doc)} | Documentos por autor: {total_docs} "
          f"| Média por autor: {media:.1f}")


def configurar_argumentos() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--documents", required=True, type=Path,
                        help="Caminho para filtered_documents.json em formato JSONL.")
    parser.add_argument("--qrels", required=True, type=Path,
                        help="Caminho para LExR-prof-qrels_filtrado (TSV).")
    parser.add_argument("--output-authors", type=Path, default=Path("perfis_autores.json"),
                        help="Saída agregada por autor.")
    parser.add_argument("--output-documents", type=Path, default=Path("perfis_por_documento.json"),
                        help="Saída separada por documento para Coverage@k.")
    parser.add_argument("--workers", type=int, default=max(2, cpu_count() - 1),
                        help="Processos paralelos (1 = execução sequencial).")
    parser.add_argument("--batch-spacy", type=int, default=BATCH_SPACY,
                        help="Tamanho de lote do spaCy (padrão 128).")
    parser.add_argument("--mode", choices=("ambos", "autores", "documentos"),
                        default="ambos", help="Quais perfis gerar (padrão: ambos).")
    return parser


def main() -> None:
    args = configurar_argumentos().parse_args()
    if args.workers < 1:
        raise ValueError("--workers deve ser um inteiro maior ou igual a 1")
    if args.batch_spacy < 1:
        raise ValueError("--batch-spacy deve ser um inteiro maior ou igual a 1")
    for caminho in (args.documents, args.qrels):
        if not caminho.is_file():
            raise FileNotFoundError(f"Arquivo de entrada não encontrado: {caminho}")

    autores_alvo = carregar_autores_qrels(args.qrels)
    print(f">> Autores encontrados no ground truth: {len(autores_alvo)}")
    if args.mode in ("ambos", "autores"):
        gerar_perfis_autores(
            args.documents, autores_alvo, args.output_authors,
            args.workers, args.batch_spacy,
        )
    if args.mode in ("ambos", "documentos"):
        gerar_perfis_documentos(
            args.documents, autores_alvo, args.output_documents,
            args.workers, args.batch_spacy,
        )


if __name__ == "__main__":
    main()
