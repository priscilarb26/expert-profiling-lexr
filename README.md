## Expert Profiling com Modelos de Linguagem e Recuperação de Informação

Este repositório contém os códigos, experimentos e resultados desenvolvidos no Trabalho de Conclusão de Curso **“Perfilamento de especialistas: as tags geradas por modelos de linguagem convergem com o julgamento humano de expertise?”**, desenvolvido no curso de Pós-Graduação em Ciência de Dados da Universidade Federal de Minas Gerais (UFMG).

O projeto investiga o uso de métodos de Recuperação de Informação e Modelos de Linguagem na tarefa de **Expert Profiling**, cujo objetivo é identificar automaticamente as principais áreas de expertise de um pesquisador a partir de suas publicações científicas.

A avaliação compara as tags geradas automaticamente com os julgamentos humanos disponíveis na coleção **LExR — Lattes Expertise Retrieval**.

---

## Problema de pesquisa

O trabalho busca responder à seguinte questão:

> Em que medida as tags de expertise geradas por modelos de linguagem convergem com o julgamento humano de expertise registrado no LExR?

Para investigar essa questão, foram comparadas abordagens lexicais tradicionais e modelos de linguagem da família Qwen, considerando diferentes estratégias de geração e ajuste.

---

## Visão geral do pipeline

O fluxo experimental pode ser resumido da seguinte forma:

```text
Publicações dos pesquisadores
        │
        ▼
Construção dos perfis
        │
        ├───────────────┐
        │               │
        ▼               ▼
     POP / TF-IDF     Modelos Qwen
                        │
              ┌─────────┼─────────┐
              ▼         ▼         ▼
          Zero-shot  Few-shot  Fine-tuning
                                 QLoRA
                        │
                        ▼
                Tags de expertise
                        │
                        ▼
              Similaridade semântica
                     SBERT
                        │
                        ▼
             Pareamento 1-para-1
                        │
                        ▼
                  Avaliação
```

---

## Base de dados

Os experimentos utilizam a coleção **LExR (Lattes Expertise Retrieval)**, construída a partir de informações da Plataforma Lattes.

Após o pré-processamento e a remoção de pesquisadores sem publicações disponíveis, o conjunto utilizado no projeto contém:

* **1.431 pesquisadores**
* **406.764 publicações**
* título, palavras-chave e resumo como fontes de evidência textual
* julgamentos humanos de expertise com relevância graduada de **1 a 3**

Para os experimentos de fine-tuning, os pesquisadores foram divididos em:

* **70% treino — 1.001 autores**
* **15% validação — 215 autores**
* **15% teste — 215 autores**

A divisão foi realizada no nível de autor utilizando uma semente fixa, garantindo que não haja sobreposição de pesquisadores entre os conjuntos.

> A base original LExR não é incluída integralmente neste repositório. A pasta [`data/`](data/) contém os arquivos derivados necessários à organização e reprodução dos experimentos.

---

## Abordagens avaliadas

### Baselines

Foram utilizadas duas abordagens clássicas de Recuperação de Informação:

* **POP**, baseado na frequência dos termos no perfil do pesquisador
* **TF-IDF**, que combina frequência local e capacidade discriminativa dos termos na coleção

### Modelos de linguagem

Foram avaliadas diferentes variantes da família Qwen:

* Qwen2.5-1.5B
* Qwen2.5-3B
* Qwen3-1.7B
* Qwen3-4B
* Qwen3.5-2B
* Qwen3.5-4B

A comparação inicial entre os modelos foi realizada em configuração **few-shot** sobre o conjunto completo de pesquisadores.

A partir dessa etapa, os modelos **Qwen3-1.7B** e **Qwen3.5-4B** também foram avaliados nas configurações:

* zero-shot
* few-shot
* fine-tuning com QLoRA

---

## Avaliação semântica

Como os modelos de linguagem podem produzir tags semanticamente equivalentes às anotações do LExR utilizando palavras diferentes, a avaliação não depende apenas de correspondência lexical exata.

O pipeline de avaliação utiliza:

1. embeddings com **Sentence-BERT**
2. similaridade de cosseno
3. limiar de similaridade de **0,75**
4. correspondência 1-para-1 entre tags preditas e tags do ground truth
5. atribuição da relevância associada à anotação correspondente

Quando nenhuma correspondência válida é encontrada, a tag recebe relevância igual a zero.

---

## Métricas

As abordagens são avaliadas utilizando métricas de recuperação, ranqueamento e características do perfil gerado:

* nDCG@10 
* MAP@10 
* Precision@10
* Recall@10 
* Coverage@10
* Diversity@10 
* proporção de matches válidos

---

## Principais resultados

Os experimentos mostraram que não houve uma vantagem consistente dos modelos de linguagem sobre as abordagens lexicais em todas as condições avaliadas.

No conjunto completo de **1.431 pesquisadores**, o **Qwen3.5-4B em configuração few-shot** apresentou o melhor desempenho global entre as abordagens avaliadas.

No conjunto de teste de **215 pesquisadores**, utilizado para a comparação controlada entre todas as configurações, o **TF-IDF apresentou os maiores valores nas principais métricas de recuperação e ranqueamento**.

Entre as configurações dos modelos de linguagem, o **few-shot apresentou melhor desempenho que o zero-shot nas principais métricas**, enquanto o fine-tuning com QLoRA não produziu ganhos consistentes em relação ao few-shot.

Os resultados indicam que métodos lexicais continuam competitivos para a tarefa de Expert Profiling, enquanto os modelos de linguagem apresentam características complementares, principalmente por sua capacidade de produzir vocabulário aberto e semanticamente variado.

---

## Experimentos complementares

Além da comparação principal apresentada no TCC, o repositório contém análises complementares relacionadas ao comportamento dos modelos e do protocolo de avaliação.

Entre elas estão:

* análise por área do conhecimento
* análise de sensibilidade do limiar de similaridade
* análise de parâmetros de geração
* comparação de `do_sample=True` e `do_sample=False`
* análise exploratória de `repetition_penalty`

---

## Testes com currículos estruturados

O repositório também contém uma extensão experimental que investiga a geração de perfis de expertise diretamente a partir de **currículos estruturados**.

Essa etapa explora a aplicação do pipeline desenvolvido no TCC em uma representação curricular mais ampla do pesquisador.

Foram realizados experimentos utilizando:

* TF-IDF sobre currículos estruturados
* Qwen3.5-4B sobre currículos estruturados
* comparação das tags produzidas com o mesmo ground truth do LExR

Esses experimentos são disponibilizados como uma **extensão prática do trabalho** e não fazem parte da comparação experimental principal utilizada para responder à questão de pesquisa do TCC.
```
## Estrutura do repositório

├── README.md
├── reprodutibilidade.md
├── requirements.txt
│
├── data/
│   ├── processed/
│   ├── curriculos/
│   └── samples/
│
├── src/
│   ├── preprocessing/
│   ├── baselines/
│   ├── llm/
│   ├── finetuning/
│   └── evaluation/
│
├── experiments/
│   ├── 01_preprocessing/
│   ├── 02_baselines/
│   ├── 03_qwen_few_shot/
│   ├── 04_zero_shot/
│   ├── 05_finetuning/
│   ├── 06_additional_analysis/
│   └── 07_curriculum_tests/│   
│
├── prompts/
├── results/
└── figures/
```

---

## Reprodutibilidade

Os detalhes necessários para reprodução dos experimentos estão documentados separadamente em:

**[`reprodutibilidade.md`](reprodutibilidade.md)**

Esse documento apresenta, entre outros pontos:

* ambiente computacional
* versões das bibliotecas
* modelos utilizados
* parâmetros de geração
* prompts
* limites de contexto
* configuração do QLoRA
* seed utilizada
* procedimento de avaliação
* parâmetros de correspondência semântica
* ordem de execução dos experimentos
* arquivos de entrada e saída

Essa separação mantém o README direcionado à apresentação geral do projeto, enquanto os detalhes técnicos permanecem documentados de forma completa.

---

## Tecnologias utilizadas

O projeto utiliza principalmente:

* Python
* PyTorch
* Hugging Face Transformers
* Sentence Transformers
* scikit-learn
* spaCy
* pandas
* NumPy
* SciPy
* Google Colab (NVIDIA A100)

---

## Como executar

A reprodução completa envolve diferentes etapas do pipeline.

Primeiro, instale as dependências:

```bash
pip install -r requirements.txt
```

Em seguida, prepare os arquivos descritos em:

```text
data/
```

A ordem completa de execução dos experimentos e os parâmetros necessários estão disponíveis em:

```text
reprodutibilidade.md
```

---

## Trabalho acadêmico

Este projeto foi desenvolvido como Trabalho de Conclusão do curso de Pós-Graduação em Ciência de Dados da **Universidade Federal de Minas Gerais — UFMG**.

**Autora:** Priscila Borges 

**Orientador:** Dr. Rodrygo Luis Teodoro Santos

**Ano:** 2026
