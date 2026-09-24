# Código-fonte

Esta pasta contém os módulos responsáveis pelo processamento, treinamento e
avaliação dos modelos utilizados no projeto de perfilamento de especialistas
sobre a base LExR.

## Organização

### `preprocessing/`

Módulos relacionados ao pré-processamento dos dados, incluindo:

- leitura e organização dos dados da base LExR;
- limpeza e normalização dos textos;
- preparação dos exemplos para treinamento e avaliação;
- criação dos formatos de entrada utilizados pelos modelos.

### `baselines/`

Implementações dos métodos de referência utilizados como baselines lexicais,
como abordagens baseadas em:

- frequência de termos;
- similaridade textual;
- correspondência entre competências e descrições;
- outras heurísticas lexicais.

### `llm/`

Código relacionado ao uso de modelos de linguagem, especialmente os modelos
Qwen. Pode incluir:

- construção de prompts;
- inferência;
- carregamento de modelos e tokenizadores;
- geração de perfis ou classificações;
- configuração dos parâmetros de execução.

### `finetuning/`

Scripts e configurações para o fine-tuning dos modelos de linguagem sobre os
dados do projeto.

Nesta pasta podem ser encontrados recursos para:

- preparação dos conjuntos de treinamento;
- configuração do processo de fine-tuning;
- treinamento dos modelos;
- salvamento e carregamento de checkpoints.

### `evaluation/`

Rotinas para avaliação dos resultados produzidos pelos baselines e pelos
modelos de linguagem.

As avaliações podem incluir:

- métricas de classificação ou ranqueamento;
- comparação entre predições e respostas esperadas;
- avaliação semântica;
- geração de tabelas e relatórios;
- comparação entre diferentes experimentos.

## Fluxo geral

De forma simplificada, o fluxo de execução do projeto é:

```text
Dados da base LExR
        |
        v
preprocessing/
        |
        +--> baselines/
        |
        +--> llm/
                |
                +--> finetuning/
        |
        v
evaluation/
        |
        v
Resultados e métricas
```

## Execução

Os comandos devem ser executados a partir da raiz do repositório:

```bash
cd expert-profiling-lexr
```

## Organização recomendada dos resultados

Para facilitar a reprodução dos experimentos, recomenda-se separar os
resultados por abordagem e configuração:

```text
results/
├── baselines/
├── prompting/
├── finetuning/
└── evaluation/
```

## Reprodutibilidade

Antes de executar os experimentos:

1. configure o ambiente Python descrito no README principal;
2. verifique se os dados necessários estão disponíveis;
3. confirme os caminhos dos arquivos de entrada;
4. registre os parâmetros utilizados;
5. salve os resultados e métricas gerados.

