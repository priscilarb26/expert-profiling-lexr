# Detalhes de implementação e reprodutibilidade

Este documento reúne os detalhes operacionais dos experimentos desenvolvidos no trabalho sobre perfilamento de especialistas. O objetivo é complementar a descrição metodológica apresentada no TCC e registrar as informações necessárias para compreender e reproduzir os experimentos disponibilizados neste repositório.

São documentados o ambiente de software, as principais versões de bibliotecas, a organização do código, o carregamento dos modelos Qwen, as configurações de inferência, os prompts utilizados, os módulos empregados no ajuste fino com QLoRA, a estratégia de contingência e retomada da inferência, o protocolo comum de avaliação e as análises complementares.

## 1. Estrutura do repositório

| Caminho no repositório | Finalidade |
|---|---|
| `src/preprocessing/` | Pré-processamento do LExR e construção dos perfis utilizados pelos baselines e pelos modelos Qwen |
| `src/baselines/pop.py` | Implementação reutilizável do baseline POP |
| `src/baselines/tfidf.py` | Implementação reutilizável do baseline TF-IDF |
| `src/llm/` | Inferência genérica dos modelos Qwen, construção dos prompts e parsing das tags |
| `src/finetuning/` | Preparação do dataset, treinamento QLoRA e inferência dos modelos ajustados |
| `src/evaluation/semantic_matching.py` | Similaridade semântica, matrizes de similaridade e matching global greedy 1-para-1 |
| `src/evaluation/metrics.py` | Cálculo das métricas de avaliação |
| `experiments/01_preprocessing/` | Notebooks de preparação do LExR e construção dos perfis |
| `experiments/02_baselines/` | Execução de POP e TF-IDF |
| `experiments/03_qwen_few_shot/` | Execução das seis variantes Qwen em few-shot no conjunto completo de 1.431 autores |
| `experiments/04_zero_shot/` | Execução zero-shot do Qwen3-1.7B e Qwen3.5-4B no conjunto de teste de 215 autores |
| `experiments/05_fine_tuning/` | Ajuste fino e inferência dos modelos Qwen3-1.7B e Qwen3.5-4B |
| `experiments/06_additional_analysis/` | Sensibilidade dos limiares, análise por área e teste com `repetition_penalty=1.1` |
| `experiments/07_curriculum_tests/` | Experimentos suplementares com currículos estruturados |
| `prompts/` | Textos literais dos prompts zero-shot, few-shot e fine-tuning |
| `results/` | Resultados agregados, rankings, avaliações detalhadas e artefatos intermediários |
| `figures/` | Figuras finais utilizadas no trabalho |
| `requirements.txt` | Dependências do projeto |

### 1.1 Notebooks de pré-processamento

- `experiments/01_preprocessing/01_prepare_lexr.ipynb`
- `experiments/01_preprocessing/02_build_profiles_baseline.ipynb`
- `experiments/01_preprocessing/03_build_profiles_qwen.ipynb`
- `experiments/01_preprocessing/04_build_document_profiles_qwen.ipynb`

### 1.2 Baselines

- `experiments/02_baselines/01_pop.ipynb`
- `experiments/02_baselines/02_tfidf.ipynb`

### 1.3 Few-shot no conjunto completo

- `experiments/03_qwen_few_shot/01_qwen2_5_1_5b.ipynb`
- `experiments/03_qwen_few_shot/02_qwen2_5_3b.ipynb`
- `experiments/03_qwen_few_shot/03_qwen3_1_7b.ipynb`
- `experiments/03_qwen_few_shot/04_qwen3_4b.ipynb`
- `experiments/03_qwen_few_shot/05_qwen3_5_2b.ipynb`
- `experiments/03_qwen_few_shot/06_qwen3_5_4b.ipynb`

### 1.4 Zero-shot no conjunto de teste

- `experiments/04_zero_shot/01_qwen3_1_7b_zero_shot.ipynb`
- `experiments/04_zero_shot/02_qwen3_5_4b_zero_shot.ipynb`

### 1.5 Fine-tuning

- `experiments/05_fine_tuning/01_qwen3_1_7b_finetuning.ipynb`
- `experiments/05_fine_tuning/02_qwen3_5_4b_finetuning.ipynb`

### 1.6 Análises adicionais

- `experiments/06_additional_analysis/01_threshold_sensitivity.ipynb`
- `experiments/06_additional_analysis/02_analysis_by_area.ipynb`
- `experiments/06_additional_analysis/03_repetition_penalty_1.1.ipynb`

O teste estatístico pareado de Wilcoxon é descrito no TCC, mas não foi mantido como notebook dedicado nesta organização do portfólio.

### 1.7 Testes suplementares com currículos

- `experiments/07_curriculum_tests/01_prepare_curriculum_data.ipynb`
- `experiments/07_curriculum_tests/02_tfidf_curriculum.ipynb`
- `experiments/07_curriculum_tests/03_qwen3_5_4b_curriculum.ipynb`
- `experiments/07_curriculum_tests/04_do_sample_curriculum.ipynb`

Esses testes são suplementares e não integram a comparação experimental principal baseada nas publicações do LExR.

## 2. Fluxo de reprodução dos experimentos

A reprodução principal segue a ordem abaixo.

1. Preparar os arquivos do LExR e os perfis textuais.
2. Construir os perfis de n-gramas utilizados por POP e TF-IDF.
3. Construir os perfis estruturados utilizados pelos modelos Qwen.
4. Construir os perfis documentais utilizados no cálculo de Coverage.
5. Executar POP e TF-IDF.
6. Executar as seis variantes Qwen em few-shot sobre os 1.431 autores.
7. Executar Qwen3-1.7B e Qwen3.5-4B em zero-shot sobre os 215 autores de teste.
8. Executar o ajuste fino com QLoRA dos dois modelos selecionados.
9. Executar a inferência dos modelos ajustados sobre os mesmos 215 autores.
10. Aplicar o mesmo procedimento de matching semântico e cálculo de métricas a todas as abordagens.
11. Executar, quando desejado, as análises adicionais de sensibilidade, área do conhecimento e `repetition_penalty`.
12. Executar separadamente os testes suplementares com currículos estruturados.

O split é realizado em nível de autor, com semente 42, nas proporções 70/15/15, resultando em 1.001 autores de treino, 215 de validação e 215 de teste. O conjunto de teste é fixo e deve ser reutilizado em todas as comparações realizadas sobre 215 autores.

## 3. Dados esperados

Os notebooks foram organizados para trabalhar com a seguinte convenção de arquivos processados:

- `data/processed/filtered_documents.json`
- `data/processed/ground_truth/LExR-prof-qrels_filtrado`
- `data/processed/splits/split_autores_seed42.json`
- `data/processed/perfis_estruturados_qwen.json`
- `data/processed/perfis_documento_qwen.json`

Os perfis dos baselines são produzidos em `experiments/01_preprocessing/02_build_profiles_baseline.ipynb` e os perfis utilizados pelo fine-tuning são preparados por `01_prepare_lexr.ipynb` e `src/finetuning/prepare_dataset.py`.

Os dados completos do LExR podem ser mantidos fora do GitHub quando o tamanho ou as condições de distribuição inviabilizarem a publicação direta. Nesse caso, o repositório deve manter amostras ilustrativas e instruções para regeneração dos arquivos processados.

## 4. Ambiente de software

Os experimentos foram conduzidos em Python 3.13.15 no Google Colab. A inferência e o ajuste fino supervisionado foram implementados no ecossistema Hugging Face.

A biblioteca `transformers` foi utilizada para o carregamento e a geração dos modelos Qwen, `peft` para os adaptadores QLoRA e `bitsandbytes` para quantização em 4 bits. A avaliação semântica utilizou `sentence-transformers` com o modelo `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`. O pré-processamento dos baselines empregou `spaCy`, `langdetect` e n-gramas; os notebooks de preparação também utilizam `ijson` para leitura em streaming de arquivos maiores.

Todas as instâncias de PyTorch utilizadas nos experimentos com GPU foram executadas com suporte a CUDA 12.8.

### 4.1 Versões registradas nos experimentos

| Biblioteca | Inferência Qwen | Fine-tuning Qwen3.5-4B | Fine-tuning Qwen3-1.7B | POP/TF-IDF |
|---|---:|---:|---:|---:|
| IPython | 7.34.0 | 7.34.0 | 7.34.0 | 7.34.0 |
| google | 3.0.0 | 3.0.0 | 3.0.0 | 3.0.0 |
| matplotlib | 3.10.0 | 3.11.1 | 3.10.0 | 3.10.0 |
| numpy | 2.1.3 | 2.1.3 | 2.1.3 | 2.1.3 |
| seaborn | 0.13.2 | 0.13.2 | 0.13.2 | 0.13.2 |
| sentence-transformers | 5.7.0 | 6.0.1 | 5.7.0 | 5.7.0 |
| torch | 2.11.0 | 2.11.0 | 2.11.0 | 2.11.0 |
| tqdm | 4.67.3 | 4.70.0 | 4.67.3 | 4.67.3 |
| transformers | 5.16.1 | 5.16.1 | 5.16.1 | — |
| datasets | — | 5.0.1 | 4.0.0 | — |
| pandas | — | 3.0.5 | 2.2.3 | — |
| peft | — | 0.20.0 | 0.20.0 | — |
| scikit-learn | — | 1.9.0 | 1.6.1 | — |

As versões de `spaCy`, `langdetect` e `ijson` não foram registradas no apêndice original. Por isso, elas aparecem sem pinagem estrita no `requirements.txt`.

O arquivo `requirements.txt` do repositório define um ambiente unificado para execução do código atual. Como os ambientes originais de fine-tuning possuíam diferenças entre si, a tabela acima deve ser considerada a referência para reprodução estrita de cada execução histórica.

## 5. Carregamento dos modelos Qwen

Os modelos foram carregados com `AutoModelForCausalLM` e `AutoTokenizer`, mantendo uma interface comum entre Qwen2.5, Qwen3 e Qwen3.5.

Embora o Qwen3.5 disponha de componentes multimodais, a tarefa deste trabalho utiliza exclusivamente texto. O carregamento automático seleciona a implementação textual apropriada a partir da configuração do modelo, sem necessidade de utilizar classes multimodais durante os experimentos.

## 6. Configuração de inferência

### 6.1 Parâmetros comuns

| Parâmetro | Configuração |
|---|---|
| `do_sample` | `True` no experimento principal |
| semente | `42` |
| `max_new_tokens` | `1024` |
| máximo de publicações por autor | `50` |
| contexto no few-shot do conjunto completo | `16384` tokens |
| contexto na comparação de 215 autores | `8192` tokens |
| padding | à esquerda |
| thinking em Qwen3/Qwen3.5 | `False` |

O limite de 16.384 tokens foi utilizado nas execuções few-shot sobre os 1.431 autores. Na comparação sobre o conjunto de teste, zero-shot, few-shot e fine-tuning utilizam 8.192 tokens.

### 6.2 Hiperparâmetros por modelo

| Modelo | Temperatura | Top-p | Top-k | Repetition penalty | Min-p | Thinking |
|---|---:|---:|---:|---:|---:|---|
| Qwen2.5-1.5B-Instruct | 0,7 | 0,8 | 20 | 1,1 | — | — |
| Qwen2.5-3B-Instruct | 0,7 | 0,8 | 20 | 1,05 | — | — |
| Qwen3-1.7B | 0,7 | 0,8 | 20 | 1,0 | 0 | False |
| Qwen3-4B | 0,7 | 0,8 | 20 | 1,0 | 0 | False |
| Qwen3.5-2B | 1,0 | 1,0 | 20 | 1,0 | 0 | False |
| Qwen3.5-4B | 0,7 | 0,8 | 20 | 1,0 | 0 | False |

Para Qwen2.5 foram utilizados os valores registrados nos respectivos `generation_config.json`. Para Qwen3 e Qwen3.5 foram utilizados os parâmetros empregados no modo sem raciocínio explícito.

O parâmetro `presence_penalty` não foi utilizado porque não é disponibilizado pela API de geração empregada no pipeline. Nos experimentos principais, `repetition_penalty` segue os valores indicados acima.

## 7. Prompts utilizados

Os textos literais utilizados nos experimentos estão versionados separadamente para evitar divergência entre documentação e código:

- `prompts/zero_shot.txt`
- `prompts/few_shot.txt`
- `prompts/finetuning.txt`

`src/llm/prompts.py` é responsável por carregar esses textos e inserir dinamicamente o perfil do pesquisador.

### 7.1 Zero-shot

O prompt zero-shot contém:

- mensagem de sistema definindo a tarefa de perfilamento acadêmico;
- instrução para gerar 30 tags de expertise;
- restrição de 1 a 3 palavras por tag;
- preservação do idioma das publicações;
- uso exclusivo das evidências fornecidas;
- resposta obrigatória em JSON;
- marcador `{{PUBLICACOES_AUTOR_JSON}}` para inserção das publicações do autor.

O texto literal está em `prompts/zero_shot.txt` e esse arquivo deve ser tratado como fonte de verdade do experimento.

### 7.2 Few-shot

O prompt few-shot reutiliza a mesma instrução principal, acrescida de dois exemplos de demonstração, um predominantemente em inglês e outro em português. O perfil real é inserido no marcador `{{PUBLICACOES_AUTOR_JSON}}`.

O texto literal está em `prompts/few_shot.txt`.

### 7.3 Fine-tuning

O prompt de fine-tuning solicita a geração de uma lista JSON de tags a partir do perfil textual do pesquisador. Durante a preparação do alvo supervisionado, as tags do LExR são ordenadas por relevância 3, depois 2, depois 1.

O marcador utilizado para inserir o conteúdo do autor é `{{PROFILE_TEXT}}`.

O texto literal está em `prompts/finetuning.txt`.

## 8. Ajuste fino com QLoRA

O ajuste fino foi aplicado ao Qwen3-1.7B e ao Qwen3.5-4B.

Os adaptadores foram inseridos nas projeções lineares comuns do decodificador textual:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

As projeções específicas da arquitetura híbrida/multimodal do Qwen3.5 não foram adaptadas, preservando um conjunto comum de módulos-alvo entre os dois modelos.

### 8.1 Hiperparâmetros QLoRA

| Parâmetro | Valor |
|---|---:|
| rank LoRA (`r`) | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0,05 |
| learning rate | `2e-4` |
| épocas máximas | 3 |
| quantização | 4 bits |

O particionamento utiliza 1.001 autores para treinamento, 215 para validação e 215 para teste. A inferência final dos modelos ajustados é feita apenas sobre os 215 autores de teste.

## 9. Contingência e retomada da inferência

A inferência possui salvamento periódico em checkpoint para reduzir a perda de processamento em caso de interrupção do ambiente Colab.

Quando uma execução é retomada, autores já concluídos são preservados e a geração continua a partir do último progresso salvo. Como a semente é estabelecida no início do processo e `do_sample=True` é utilizado no experimento principal, uma retomada pode alterar a sequência posterior de amostragem em relação a uma execução contínua. Por essa razão, os arquivos de checkpoint são artefatos de contingência, não resultados científicos finais.

## 10. Protocolo comum de avaliação

Todas as abordagens produzem uma lista ordenada de tags por autor e são submetidas ao mesmo procedimento de avaliação.

### 10.1 Correspondência semântica

- modelo: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`;
- vetores normalizados por L2;
- similaridade de cosseno;
- matching global greedy 1-para-1;
- threshold principal `theta = 0.75`;
- uma tag do ground truth pode ser associada a no máximo uma predição;
- predições sem match válido recebem relevância 0.

O matching não reordena as tags: o ranking avaliado continua sendo a ordem gerada pela abordagem original.

### 10.2 Métricas

São calculadas:

- nDCG@10 e nDCG@20;
- Precision@5, @10 e @20;
- Recall@5, @10 e @20;
- MAP@10 e MAP@20;
- Coverage@10 e Coverage@20;
- Diversity@10 e Diversity@20;
- Match_Valido@10 e Match_Valido@20.

Para Precision, Recall e MAP, são consideradas relevantes as anotações do LExR com peso maior ou igual a 2. O nDCG preserva os graus 1, 2 e 3.

### 10.3 Coverage

Coverage utiliza `theta_cov = 0.75`.

Nos baselines, a correspondência exata é verificada contra os n-gramas da publicação. Nos modelos Qwen, a unidade documental preserva título, palavras-chave e resumo. Na ausência de match exato, é aplicada a correspondência semântica.

Os perfis documentais são separados da representação usada para gerar as tags, de forma que a avaliação de Coverage não dependa do formato de entrada específico de cada modelo.

## 11. Análise de sensibilidade dos limiares

A sensibilidade foi avaliada sobre as saídas few-shot do Qwen3.5-4B.

### 11.1 Threshold de matching

| theta | nDCG@10 | R@10 | MV@10 |
|---:|---:|---:|---:|
| 0,60 | 0,5021 | 0,2055 | 5,77 |
| 0,65 | 0,4662 | 0,1882 | 5,22 |
| 0,70 | 0,4234 | 0,1683 | 4,64 |
| **0,75** | **0,3804** | **0,1490** | **4,06** |
| 0,80 | 0,3336 | 0,1281 | 3,46 |

A variação é monotônica. O valor 0,75 foi adotado como decisão operacional conservadora e não como um ótimo global determinado experimentalmente.

### 11.2 Threshold de Coverage

Com o threshold de matching fixo em 0,75:

| theta_cov | Coverage@10 | Coverage@20 |
|---:|---:|---:|
| 0,60 | 0,6151 | 0,7155 |
| 0,65 | 0,5298 | 0,6267 |
| 0,70 | 0,4526 | 0,5419 |
| **0,75** | **0,3858** | **0,4646** |
| 0,80 | 0,3294 | 0,3998 |

O mesmo valor 0,75 foi mantido para `theta` e `theta_cov` em todos os pipelines principais.

## 12. Análises adicionais

### 12.1 Análise por área do conhecimento

`experiments/06_additional_analysis/02_analysis_by_area.ipynb` contém apenas os cálculos necessários para as três figuras utilizadas no trabalho:

1. diferença de nDCG@10 por área entre Qwen3.5-4B e TF-IDF;
2. diferença de nDCG@10 por autor;
3. proporção de vitórias, empates e derrotas por área.

Os arquivos tabulares derivados ficam em `results/additional_analysis/analysis_by_area/`, enquanto as imagens finais ficam em `figures/`.

### 12.2 Repetition penalty = 1.1

`experiments/06_additional_analysis/03_repetition_penalty_1.1.ipynb` reutiliza o adaptador QLoRA treinado do Qwen3-1.7B e altera apenas o parâmetro de geração `repetition_penalty` para 1,1. O treinamento não é repetido.

## 13. Testes suplementares com currículos estruturados

Os experimentos com currículos estruturados foram mantidos separados da comparação principal com publicações.

O fluxo é:

1. transformação dos XML em representação estruturada;
2. construção dos perfis usados pelo TF-IDF;
3. construção dos perfis documentais para Coverage;
4. execução do TF-IDF;
5. execução do Qwen3.5-4B;
6. execução adicional do Qwen3.5-4B com `do_sample=False`.

Os notebooks estão em `experiments/07_curriculum_tests/` e os resultados em `results/curriculum_tests/`.

A comparação `do_sample=True` versus `do_sample=False` é tratada como análise suplementar do cenário de currículos e não como parte da comparação principal das publicações.

## 14. Resultados e artefatos intermediários

A convenção de resultados utilizada no repositório é:

- `tags_brutas*.json`: saídas/rankings produzidos pelas abordagens;
- `metricas*.csv`: métricas por autor;
- `avaliacoes_gerais*.csv`: detalhamento dos matches semânticos;
- `sim_matrices/`: matrizes `.npz` usadas para reaproveitar similaridades sem recalcular o SBERT.

As matrizes `.npz`, checkpoints, adaptadores QLoRA e caches podem ocupar espaço significativo. Quando não forem publicados no GitHub, devem ser regenerados pelos notebooks correspondentes. Os resultados agregados e os arquivos diretamente utilizados nas tabelas e figuras devem ser mantidos no repositório sempre que possível.

## 15. Instalação

Para criar o ambiente principal:

```bash
python -m pip install -r requirements.txt
```

Para o pré-processamento dos baselines, também são necessários os modelos pequenos do spaCy em português e inglês:

```bash
python -m spacy download pt_core_news_sm
python -m spacy download en_core_web_sm
```

O `requirements.txt` usa CUDA 12.8 para reproduzir o ambiente GPU original. Em uma máquina sem CUDA, a instalação do PyTorch deve ser adaptada ao ambiente local.

## 16. Observações sobre reprodução exata

- Os notebooks publicados foram reorganizados para remover caminhos privados do Google Drive e reutilizar funções presentes em `src/`.
- A lógica experimental foi preservada, mas a localização dos arquivos deve seguir a estrutura deste repositório.
- O experimento principal few-shot usa os 1.431 autores.
- Zero-shot e fine-tuning são comparados sobre o conjunto fixo de 215 autores de teste.
- Os testes com currículos são suplementares.
- Os prompts literais estão versionados em `prompts/`.
- Para reproduzir exatamente um ambiente histórico de fine-tuning, consulte a tabela de versões da Seção 4.1, pois Qwen3.5-4B e Qwen3-1.7B foram executados com algumas versões diferentes.
