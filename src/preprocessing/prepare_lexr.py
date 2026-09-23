"""Inspeção inicial e seleção de publicações da coleção LExR.

O notebook reúne inspeções exploratórias de arquivos e uma etapa efetiva de preparação:
selecionar, em ``documents.json``
(JSON Lines), as publicações de pelo menos um autor listado em
``author_unique_prof-qrels.json``. O resultado é ``documents_ft.json``
(um array JSON, como no notebook original).

Esta versão também incorpora a etapa de ``perfis_textuais_ft.ipynb``: ler
``documents_ft.json`` e gerar ``perfis_textuais_ft.json`` com um perfil por
pesquisador, unindo título, palavras-chave e resumo de cada publicação.
A ordem das publicações segue a ordem dos registros no arquivo de entrada, como no notebook.

IMPORTANTE: este script **não** cria ``filtered_documents.json``, não filtra os
qrels para os autores com publicações e não cria a partição treino/validação/
teste.  

Comandos de exemplo:

    python prepare_lexr.py filter \\
        --documents /caminho/documents.json \\
        --authors /caminho/author_unique_prof-qrels.json \\
        --output /caminho/documents_ft.json

    python prepare_lexr.py profiles-ft \\
        --documents /caminho/documents_ft.json \\
        --authors /caminho/author_unique_prof-qrels.json \\
        --output /caminho/perfis_textuais_ft.json

    python prepare_lexr.py inspect \\
        --path /caminho/LExR-prof-qrels --type qrels

    python prepare_lexr.py inspect \\
        --path /caminho/filtered_documents.json --type jsonl

    python prepare_lexr.py inspect \\
        --path /caminho/author_unique_prof-qrels.json --type ids

    python prepare_lexr.py extract \\
        --archive /caminho/LExR --output-dir /caminho/LExR_extraido
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
import stat
import tempfile
import tarfile
import zipfile


QRELS_RE = re.compile(r"^\s*(\S+)\s+(\d+)\s+(.+?)\s+(-?\d+)\s*$")
ID_KEYS = ("author_id", "autor_id", "id", "author", "autor")
LIST_KEYS = ("authors", "author_ids", "autores", "ids")


def load_author_ids(path: str | Path) -> set[str]:
    """Carrega IDs nos formatos aceitos pelo notebook de origem.

    Aceita lista de IDs, lista de objetos com ID, dicionário com lista de IDs
    ou dicionário cujas chaves são IDs. IDs são preservados como strings, para
    manter eventuais zeros à esquerda.
    """
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    ids: set[str] = set()

    def add_item(item: object) -> None:
        if isinstance(item, (str, int)) and not isinstance(item, bool):
            ids.add(str(item).strip())
        elif isinstance(item, dict):
            for key in ID_KEYS:
                if key in item and item[key] is not None:
                    ids.add(str(item[key]).strip())
                    break

    if isinstance(data, list):
        for item in data:
            add_item(item)
    elif isinstance(data, dict):
        nested_found = False
        for key in LIST_KEYS:
            if isinstance(data.get(key), list):
                for item in data[key]:
                    add_item(item)
                nested_found = True
                break
        if not nested_found:
            ids.update(str(key).strip() for key in data)
    else:
        raise ValueError(f"Formato de lista de autores não reconhecido: {source}")

    ids.discard("")
    if not ids:
        raise ValueError(f"Nenhum identificador de autor encontrado em {source}")
    return ids


def filter_documents(
    documents_path: str | Path,
    author_ids_path: str | Path,
    output_path: str | Path,
    progress_every: int = 500_000,
) -> dict[str, object]:
    """Filtra o JSONL ``documents.json`` sem carregar todas as publicações.

    Mantém uma publicação quando ``authors`` contém ao menos um dos IDs alvo.
    Salva as publicações completas como array JSON, comportamento preservado do
    notebook. Não modifica o campo ``authors`` nem os demais metadados.
    """
    documents_path = Path(documents_path)
    author_ids_path = Path(author_ids_path)
    output_path = Path(output_path)
    if documents_path.resolve() == output_path.resolve():
        raise ValueError("A saída deve ser diferente do arquivo de entrada.")

    author_ids = load_author_ids(author_ids_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    invalid_lines = 0
    kept_publications = 0
    authors_found: set[str] = set()
    publications_by_author: Counter[str] = Counter()

    print(f"Autores-alvo carregados: {len(author_ids):,}")
    print(f"Entrada: {documents_path}")
    print(f"Saída:   {output_path}")

    # Arquivo temporário reduz o risco de deixar uma saída parcial se houver
    # interrupção; a gravação permanece streaming.
    temporary_path = output_path.with_name(output_path.name + ".partial")
    try:
        with documents_path.open("r", encoding="utf-8") as src, temporary_path.open(
            "w", encoding="utf-8"
        ) as dst:
            dst.write("[\n")
            first = True
            for line in src:
                total_lines += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    publication = json.loads(line)
                except json.JSONDecodeError:
                    invalid_lines += 1
                    continue

                if not isinstance(publication, dict):
                    continue
                publication_authors = publication.get("authors", [])
                if not isinstance(publication_authors, list):
                    continue
                selected = {
                    str(author).strip()
                    for author in publication_authors
                    if str(author).strip() in author_ids
                }
                if selected:
                    if not first:
                        dst.write(",\n")
                    json.dump(publication, dst, ensure_ascii=False)
                    first = False
                    kept_publications += 1
                    authors_found.update(selected)
                    publications_by_author.update(selected)

                if progress_every > 0 and total_lines % progress_every == 0:
                    print(
                        f"Lidas: {total_lines:,} | Mantidas: {kept_publications:,} | "
                        f"Autores encontrados: {len(authors_found):,}"
                    )
            dst.write("\n]\n")
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    missing_authors = author_ids - authors_found
    summary: dict[str, object] = {
        "total_linhas": total_lines,
        "linhas_invalidas": invalid_lines,
        "publicacoes_mantidas": kept_publications,
        "autores_alvo": len(author_ids),
        "autores_encontrados": len(authors_found),
        "autores_sem_publicacao": len(missing_authors),
        "exemplos_autores_sem_publicacao": sorted(missing_authors)[:10],
    }
    print("\nProcessamento concluído:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary



def _clean_field(value: object) -> str:
    """Conversão utilizada no notebook: listas separadas por '; ', sem outras normalizações."""
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(
            str(item).strip()
            for item in value
            if item is not None and str(item).strip()
        )
    return str(value).strip()


def _publication_text(publication: dict[str, object]) -> str:
    """Preserva rótulos e a ordem título -> palavras-chave -> resumo do notebook."""
    title = _clean_field(publication.get("title", ""))
    keywords = _clean_field(publication.get("keywords", ""))
    abstract = _clean_field(publication.get("abstract", ""))
    parts = []
    if title:
        parts.append(f"Título: {title}")
    if keywords:
        parts.append(f"Palavras-chave: {keywords}")
    if abstract:
        parts.append(f"Resumo: {abstract}")
    return "\n".join(parts)


def _iter_json_array_stdlib(path: Path, chunk_size: int = 1024 * 1024):
    """Lê um array JSON em fluxo com a biblioteca padrão, sem carregar todo o arquivo.

    Equivale a ``ijson.items(f, 'item')`` para o array de documentos. Se ijson
    estiver disponível, ``build_ft_text_profiles`` o utiliza diretamente.
    """
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as stream:
        buffer = ""
        pos = 0
        eof = False

        def advance() -> None:
            nonlocal buffer, pos, eof
            buffer = buffer[pos:]
            pos = 0
            incoming = stream.read(chunk_size)
            if incoming:
                buffer += incoming
            else:
                eof = True

        def skip_whitespace() -> None:
            nonlocal pos
            while True:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos < len(buffer) or eof:
                    break
                advance()

        advance()
        skip_whitespace()
        if pos >= len(buffer) or buffer[pos] != "[":
            raise ValueError(f"Esperado array JSON de documentos em {path}")
        pos += 1
        skip_whitespace()

        if pos < len(buffer) and buffer[pos] == "]":
            pos += 1
        else:
            while True:
                skip_whitespace()
                while True:
                    try:
                        item, end_pos = decoder.raw_decode(buffer, pos)
                        pos = end_pos
                        break
                    except json.JSONDecodeError as exc:
                        if eof:
                            raise ValueError(f"JSON inválido em {path}: {exc}") from exc
                        advance()
                yield item
                skip_whitespace()
                if pos >= len(buffer):
                    raise ValueError(f"Array JSON incompleto em {path}")
                separator = buffer[pos]
                pos += 1
                if separator == "]":
                    break
                if separator != ",":
                    raise ValueError(f"Separador inválido no array JSON de {path}: {separator!r}")

        skip_whitespace()
        if pos < len(buffer) or not eof:
            # Verifica se só há espaços depois do fechamento do array.
            rest = buffer[pos:] + stream.read()
            if rest.strip():
                raise ValueError(f"Conteúdo extra após o array JSON em {path}")


def _iter_ft_documents(path: Path):
    """Usa o leitor de fluxo original ijson, com fallback sem dependência externa."""
    try:
        import ijson
    except ImportError:
        yield from _iter_json_array_stdlib(path)
    else:
        with path.open("rb") as handle:
            yield from ijson.items(handle, "item")


def build_ft_text_profiles(
    documents_path: str | Path,
    author_ids_path: str | Path,
    output_path: str | Path,
    temp_dir: str | Path | None = None,
    batch_size: int = 10_000,
    progress_every: int = 50_000,
) -> dict[str, object]:
    """Reproduz ``perfis_textuais_ft.ipynb`` após a seleção de ``documents_ft.json``.

    Gera objeto JSON {author_id: {"n_publicacoes": N, "publicacoes_textuais": []}},
    inclusive perfis vazios para autores do arquivo de referência sem documentos
    com conteúdo textual. Não limita a quantidade de publicações nem altera
    seus textos além de remover espaços nas extremidades de cada campo.
    """
    documents_path = Path(documents_path)
    author_ids_path = Path(author_ids_path)
    output_path = Path(output_path)
    if documents_path.resolve() == output_path.resolve():
        raise ValueError("A saída deve ser diferente do arquivo de entrada.")
    if author_ids_path.resolve() == output_path.resolve():
        raise ValueError("A saída não pode sobrescrever o arquivo de autores.")
    if batch_size < 1:
        raise ValueError("O tamanho do lote deve ser positivo.")

    author_ids = load_author_ids(author_ids_path)
    print(f"Autores-alvo carregados: {len(author_ids):,}")
    print(f"Entrada: {documents_path}")
    print(f"Saída:   {output_path}")

    total_publications = 0
    publications_with_text = 0
    author_document_links = 0
    found_authors: set[str] = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(output_path.name + ".partial")
    try:
        # TemporaryDirectory remove o banco mesmo em caso de erro/interrupção.
        with tempfile.TemporaryDirectory(dir=temp_dir, prefix="lexr_profiles_") as tmp:
            connection = sqlite3.connect(Path(tmp) / "profiles.sqlite")
            try:
                connection.execute(
                    "CREATE TABLE texts_by_author (author_id TEXT NOT NULL, "
                    "position INTEGER NOT NULL, text TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE INDEX idx_texts_author ON texts_by_author(author_id)"
                )
                buffer: list[tuple[str, int, str]] = []
                for position, publication in enumerate(_iter_ft_documents(documents_path), 1):
                    total_publications += 1
                    if not isinstance(publication, dict):
                        raise ValueError(f"Documento {position} não é um objeto JSON.")
                    text_value = _publication_text(publication)
                    if not text_value:
                        continue
                    authors = publication.get("authors", [])
                    if not isinstance(authors, list):
                        continue
                    selected = {
                        str(author).strip()
                        for author in authors
                        if str(author).strip() in author_ids
                    }
                    if not selected:
                        continue
                    publications_with_text += 1
                    for author_id in selected:
                        buffer.append((author_id, position, text_value))
                        found_authors.add(author_id)
                        author_document_links += 1
                    if len(buffer) >= batch_size:
                        connection.executemany(
                            "INSERT INTO texts_by_author VALUES (?, ?, ?)", buffer
                        )
                        connection.commit()
                        buffer.clear()
                    if progress_every > 0 and total_publications % progress_every == 0:
                        print(
                            f"Publicações lidas: {total_publications:,} | "
                            f"Autores encontrados: {len(found_authors):,} | "
                            f"Vínculos autor-publicação: {author_document_links:,}"
                        )
                if buffer:
                    connection.executemany(
                        "INSERT INTO texts_by_author VALUES (?, ?, ?)", buffer
                    )
                    connection.commit()

                with temporary_output.open("w", encoding="utf-8") as out:
                    out.write("{\n")
                    for index, author_id in enumerate(sorted(author_ids)):
                        rows = connection.execute(
                            "SELECT text FROM texts_by_author WHERE author_id = ? "
                            "ORDER BY position", (author_id,)
                        )
                        texts = [row[0] for row in rows]
                        if index:
                            out.write(",\n")
                        out.write(json.dumps(author_id, ensure_ascii=False))
                        out.write(": ")
                        json.dump(
                            {"n_publicacoes": len(texts), "publicacoes_textuais": texts},
                            out,
                            ensure_ascii=False,
                        )
                    out.write("\n}\n")
                temporary_output.replace(output_path)
            finally:
                connection.close()
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        raise

    missing_authors = author_ids - found_authors
    summary: dict[str, object] = {
        "autores_referencia": len(author_ids),
        "autores_com_publicacao_textual": len(found_authors),
        "autores_sem_publicacao_textual": len(missing_authors),
        "exemplos_sem_publicacao": sorted(missing_authors)[:10],
        "publicacoes_lidas": total_publications,
        "publicacoes_com_texto_aproveitavel": publications_with_text,
        "vinculos_autor_publicacao": author_document_links,
        "arquivo_saida": str(output_path),
    }
    print("\nPerfis textuais preparados para fine-tuning:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def inspect_ids(path: str | Path, examples: int = 3) -> None:
    """Resume arquivo de IDs em lista JSON, sem modificar os dados."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Esperada uma lista JSON: {path}")
    print(f"Arquivo: {path}")
    print(f"Total de IDs: {len(data):,}")
    # No notebook de inspeção espera-se uma lista simples de IDs.
    unique_ids = {str(item) for item in data}
    print(f"IDs únicos: {len(unique_ids):,}")
    print(f"Repetidos: {len(data) - len(unique_ids):,}")
    for i, item in enumerate(data[:examples], 1):
        print(f"  {i}: {item}")


def inspect_documents_jsonl(path: str | Path, examples: int = 3) -> None:
    """Inspeciona um JSONL de documentos, mantendo uso de memória reduzido."""
    path = Path(path)
    total, with_id, with_authors, total_authorships = 0, 0, 0, 0
    key_counts: Counter[str] = Counter()
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL inválido em {path}, linha {line_number}: {exc}") from exc
            if not isinstance(doc, dict):
                raise ValueError(f"Esperado objeto JSON em {path}, linha {line_number}")
            total += 1
            key_counts.update(doc)
            if doc.get("id") is not None:
                with_id += 1
            authors = doc.get("authors", [])
            if isinstance(authors, list) and authors:
                with_authors += 1
                total_authorships += len(authors)
            if len(samples) < examples:
                samples.append(doc)

    print(f"Arquivo: {path}")
    print(f"Documentos: {total:,}")
    print(f"Com id: {with_id:,}")
    print(f"Com autores: {with_authors:,}")
    print(f"Total de autorias: {total_authorships:,}")
    print("Frequência dos campos:")
    for name, count in key_counts.most_common():
        print(f"  {name}: {count:,} ({100 * count / total if total else 0:.2f}%)")
    print("Exemplos:")
    for item in samples:
        print(json.dumps(item, ensure_ascii=False, indent=2))


def inspect_qrels(path: str | Path, examples: int = 3, ignore_invalid: bool = False) -> None:
    """Inspeciona julgamentos LExR; tags podem conter espaços."""
    path = Path(path)
    authors, tags = set(), set()
    distribution: Counter[int] = Counter()
    valid, invalid = 0, 0
    samples = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.replace("\x00", "").strip()
            if not line:
                continue
            match = QRELS_RE.match(line)
            if match is None:
                invalid += 1
                if ignore_invalid:
                    continue
                raise ValueError(f"Linha qrels inválida em {path}:{line_number}: {line[:120]}")
            author_id, iteration, tag, relevance = match.groups()
            valid += 1
            authors.add(author_id)
            tags.add(tag)
            distribution[int(relevance)] += 1
            if len(samples) < examples:
                samples.append((author_id, iteration, tag, int(relevance)))
    print(f"Arquivo: {path}")
    print(f"Julgamentos válidos: {valid:,}; linhas inválidas: {invalid:,}")
    print(f"Autores únicos: {len(authors):,}; tags únicas: {len(tags):,}")
    print(f"Distribuição de relevância: {dict(sorted(distribution.items()))}")
    for sample in samples:
        print(f"  autor={sample[0]} iteração={sample[1]} tag={sample[2]!r} relevância={sample[3]}")


def _safe_target(destination: Path, archive_member_name: str) -> Path:
    """Impede que membros de arquivos compactados escapem da pasta destino."""
    normalized = archive_member_name.replace("\\", "/")
    name = PurePosixPath(normalized)
    if name.is_absolute() or ".." in name.parts or re.match(r"^[a-zA-Z]:", normalized):
        raise ValueError(f"Caminho inseguro no arquivo compactado: {archive_member_name!r}")
    target = destination.joinpath(*name.parts)
    if not target.resolve().is_relative_to(destination.resolve()):
        raise ValueError(f"Caminho fora do destino: {archive_member_name!r}")
    return target


def extract_archive(archive_path: str | Path, output_dir: str | Path) -> None:
    """Extrai ZIP/TAR para uma pasta, recusando caminhos perigosos e links."""
    archive_path, output_dir = Path(archive_path), Path(output_dir)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                _safe_target(output_dir, member.filename)
                mode = (member.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    raise ValueError(f"Link simbólico não permitido: {member.filename}")
            output_dir.mkdir(parents=True, exist_ok=True)
            archive.extractall(output_dir)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as archive:
            members = archive.getmembers()
            for member in members:
                _safe_target(output_dir, member.name)
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f"Tipo de membro TAR não permitido: {member.name}")
            output_dir.mkdir(parents=True, exist_ok=True)
            for member in members:
                target = _safe_target(output_dir, member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError(f"Não foi possível ler {member.name}")
                    with stream, target.open("wb") as dest:
                        import shutil
                        shutil.copyfileobj(stream, dest)
    else:
        raise ValueError(f"Arquivo não reconhecido como ZIP ou TAR: {archive_path}")
    print(f"Extração concluída em: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    filtering = commands.add_parser("filter", help="Filtrar documentos JSONL pelos autores dos qrels")
    filtering.add_argument("--documents", type=Path, required=True, help="documents.json (formato JSONL)")
    filtering.add_argument("--authors", type=Path, required=True, help="author_unique_prof-qrels.json")
    filtering.add_argument("--output", type=Path, required=True, help="documents_ft.json (array JSON)")
    filtering.add_argument("--progress-every", type=int, default=500_000)

    profiles = commands.add_parser(
        "profiles-ft", help="Gerar perfis textuais por autor a partir de documents_ft.json"
    )
    profiles.add_argument("--documents", type=Path, required=True, help="documents_ft.json (array JSON)")
    profiles.add_argument("--authors", type=Path, required=True, help="author_unique_prof-qrels.json")
    profiles.add_argument("--output", type=Path, required=True, help="perfis_textuais_ft.json")
    profiles.add_argument("--temp-dir", type=Path, default=None, help="Diretório para SQLite temporário")
    profiles.add_argument("--batch-size", type=int, default=10_000)
    profiles.add_argument("--progress-every", type=int, default=50_000)

    inspection = commands.add_parser("inspect", help="Inspecionar formatos da base sem modificá-los")
    inspection.add_argument("--path", type=Path, required=True)
    inspection.add_argument("--type", choices=("ids", "jsonl", "qrels"), required=True)
    inspection.add_argument("--examples", type=int, default=3)
    inspection.add_argument("--ignore-invalid", action="store_true", help="Ignorar linhas qrels fora do padrão")

    extraction = commands.add_parser("extract", help="Extrair um ZIP/TAR da coleção LExR")
    extraction.add_argument("--archive", type=Path, required=True)
    extraction.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "filter":
        filter_documents(args.documents, args.authors, args.output, args.progress_every)
    elif args.command == "profiles-ft":
        build_ft_text_profiles(
            args.documents,
            args.authors,
            args.output,
            temp_dir=args.temp_dir,
            batch_size=args.batch_size,
            progress_every=args.progress_every,
        )
    elif args.command == "inspect":
        if args.type == "ids":
            inspect_ids(args.path, args.examples)
        elif args.type == "jsonl":
            inspect_documents_jsonl(args.path, args.examples)
        else:
            inspect_qrels(args.path, args.examples, args.ignore_invalid)
    elif args.command == "extract":
        extract_archive(args.archive, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
