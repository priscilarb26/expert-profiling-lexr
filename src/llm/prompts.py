"""Prompts bilíngues

O ``tokenizer.apply_chat_template`` pertence à etapa de inferência, permitindo
usar o template próprio de cada família Qwen. A variante ``zero-shot`` é uma
extensão organizacional: usa o mesmo pedido real e não inclui as demonstrações.

A seleção ``publicacoes[:max_publicacoes]`` mantém a ordem do JSON recebido;
este módulo não ordena publicações por ano.
"""

from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = (
    "Você é um sistema especialista em perfilamento acadêmico e extração de "
    "perfis de expertise. Sua tarefa é analisar as publicações de um autor e "
    "extrair as principais áreas de pesquisa em forma de tags. "
    "Responda APENAS com um JSON contendo a lista de tags, sem texto adicional, "
    "sem markdown, sem explicações."
)

# Exemplo 1 — INGLÊS
FEW_SHOT_INPUT_EN = json.dumps([
    {
        "titulo": "A machine learning approach for medical data mining",
        "keywords": "machine learning, data mining, medical informatics",
        "abstract": "This paper presents a novel algorithm for clustering "
                    "patient records using supervised learning techniques.",
    },
    {
        "titulo": "Deep learning applied to image recognition in radiology",
        "keywords": "deep learning, neural networks, radiology",
        "abstract": "We propose a CNN architecture that improves diagnostic "
                    "accuracy on chest X-ray images.",
    },
], ensure_ascii=False, indent=2)

FEW_SHOT_OUTPUT_EN = json.dumps({
    "tags": [
        "machine learning", "deep learning", "data mining", "neural networks",
        "medical informatics", "radiology", "image recognition", "CNN",
        "supervised learning", "clustering", "patient records",
        "chest x-ray", "diagnostic accuracy", "medical imaging",
        "healthcare AI", "convolutional networks", "computer vision",
        "biomedical engineering", "pattern recognition", "classification",
    ]
}, ensure_ascii=False)

# Exemplo 2 — PORTUGUÊS
FEW_SHOT_INPUT_PT = json.dumps([
    {
        "titulo": "Química do solo em ambientes tropicais brasileiros",
        "keywords": "química do solo, solos tropicais, agronomia",
        "abstract": "Este trabalho investiga as propriedades químicas dos "
                    "solos da região amazônica, com foco em fertilidade e "
                    "manejo agrícola sustentável.",
    },
    {
        "titulo": "Ensino de química inorgânica no nível médio",
        "keywords": "ensino de química, educação básica, química inorgânica",
        "abstract": "Apresentamos uma sequência didática para o ensino de "
                    "reações de oxirredução e tabela periódica para alunos "
                    "do ensino médio em escolas públicas.",
    },
], ensure_ascii=False, indent=2)

FEW_SHOT_OUTPUT_PT = json.dumps({
    "tags": [
        "química do solo", "solos tropicais", "ensino de química",
        "química inorgânica", "agronomia", "fertilidade do solo",
        "manejo agrícola", "educação básica", "ensino médio",
        "sequência didática", "tabela periódica", "reações de oxirredução",
        "região amazônica", "química ambiental", "agricultura sustentável",
        "química educacional", "didática", "solos brasileiros",
        "química aplicada", "ciência do solo",
    ]
}, ensure_ascii=False)



def montar_mensagens(
    publicacoes_autor: list[dict[str, Any]],
    *,
    n_tags: int = 30,
    max_publicacoes: int = 50,
    modo: str = "few-shot",
) -> list[dict[str, str]]:
    """Monta mensagens no formato chat do notebook, sem aplicar chat template.

    O modo few-shot reproduz literalmente a sequência original:
    system, user EN, assistant EN, user PT, assistant PT, user real.
    """
    if modo not in {"few-shot", "zero-shot"}:
        raise ValueError("modo deve ser 'few-shot' ou 'zero-shot'.")
    if n_tags <= 0 or max_publicacoes <= 0:
        raise ValueError("n_tags e max_publicacoes devem ser positivos.")

    pubs = publicacoes_autor[:max_publicacoes]
    dados_autor_json = json.dumps(pubs, ensure_ascii=False, indent=2)

    instrucao = (
        f"Analise as publicações a seguir e extraia as {n_tags} principais "
        f"tags de expertise que melhor representam as áreas de pesquisa deste autor. "
        f"Cada tag deve ser curta (1 a 3 palavras), específica e descritiva. "
        f"IMPORTANTE: mantenha o IDIOMA das publicações nas tags — se as "
        f"publicações estão em português, as tags devem estar em português; "
        f"se estão em inglês, em inglês. "
        f"Responda APENAS com um JSON no formato: "
        f'{{"tags": ["tag1", "tag2", ...]}} — sem markdown, sem comentários.'
    )

    prefixo_ex1 = "EXEMPLO 1 (apenas demonstração do formato esperado, NÃO copie estas tags):\n\n"
    prefixo_ex2 = "EXEMPLO 2 (apenas demonstração do formato esperado, NÃO copie estas tags):\n\n"
    prefixo_real = (
        "TAREFA REAL — analise as publicações REAIS ABAIXO. "
        "NÃO repita as tags dos exemplos anteriores. "
        "Extraia tags ESPECÍFICAS para este autor, baseadas SOMENTE no "
        "conteúdo das publicações abaixo.\n\n"
    )

    mensagens = [{"role": "system", "content": SYSTEM_PROMPT}]
    if modo == "few-shot":
        mensagens.extend([
            {"role": "user", "content": prefixo_ex1 + instrucao + "\n\nPublicações:\n" + FEW_SHOT_INPUT_EN},
            {"role": "assistant", "content": FEW_SHOT_OUTPUT_EN},
            {"role": "user", "content": prefixo_ex2 + instrucao + "\n\nPublicações:\n" + FEW_SHOT_INPUT_PT},
            {"role": "assistant", "content": FEW_SHOT_OUTPUT_PT},
        ])
    mensagens.append({
        "role": "user",
        "content": prefixo_real + instrucao + "\n\nPublicações do autor:\n" + dados_autor_json,
    })
    return mensagens
