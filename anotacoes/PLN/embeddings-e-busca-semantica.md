# Embeddings e Busca Semântica — Anotações (Minilab PLN)

Anotações sobre o minilab "Busca vetorial" (materiais/PLN). Cobre embeddings, similaridade cosseno, TF-IDF, BERT, SBERT, t-SNE e recall@k.

## Conceitos base

- **Embedding**: vetor que representa o significado de um texto. Centenas de dimensões (384, 768, 1024...).
- **Similaridade cosseno**: mede o alinhamento entre dois vetores:

  ```
  cos(θ) = a·b / (||a|| · ||b||)
  ```

  - `1.0` → mesmas direção/sentido (muito parecidos)
  - `0.0` → ortogonais (sem relação)
  - `-1.0` → sentidos opostos (raro em embeddings de sentença)

- **Busca semântica** funciona em 4 passos:
  1. transformar cada texto em um vetor
  2. transformar a query em um vetor
  3. calcular a similaridade cosseno entre a query e todos os textos
  4. retornar os `top-k` mais similares

## TF-IDF (baseline)

- Cada texto vira um vetor com **uma dimensão por termo do vocabulário** (esparso, muitos zeros).
- **TF**: frequência do termo no documento. **IDF**: reduz o peso de termos muito comuns.
- Resultado: ranqueia por **sobreposição de palavras exatas**.
- **Limitação principal**: se a query usa sinônimo/paráfrase (ex.: "vector database" vs "banco de dados vetorial"), não há termos em comum → ranking ruim ou vazio. Fica **preso às palavras**.

## BERT "puro" (transformers)

- O BERT devolve **um vetor por token**: saída `(batch, seq_len, hidden_size)`, ex.: 768.
- Para ter **um vetor por sentença**, é preciso **pooling**.

### CLS pooling
- Pega apenas o vetor do token especial `[CLS]` (primeiro token).
- O BERT foi pré-treinado para o `[CLS]` concentrar a representação da sentença.
- Mas é **um único token**: sem fine-tuning para similaridade, nem sempre é ideal.
- Analogia: "uma palavra representa a frase".

### Mean pooling
- Tira a **média dos vetores de todos os tokens reais** (ignora padding via `attention_mask`).
- Agrega a frase inteira e é mais **estável/robusto** para similaridade.
- Analogia: "a frase inteira, em média, representa a frase".

> Detalhe do notebook: a **query** sempre é codificada com mean pooling, mesmo quando se busca contra a matriz CLS — decisão de estabilidade.

- BERT "puro" (sem treino específico) já captura **significado**, não apenas palavras → retorna resultados mais semânticos que TF-IDF.

## Sentence-BERT (SBERT) — o "jeito certo"

- Família de modelos **treinados para similaridade de sentenças**: sentenças com mesmo significado ficam com vetores próximos.
- Modelo usado: `paraphrase-multilingual-MiniLM-L12-v2` (384 dims, multilíngue, leve).
- Vantagens sobre BERT puro: embeddings já normalizados e calibrados para cosine similarity.

## Comparação prática (query com sinônimos)

- **TF-IDF**: preso às palavras exatas; um texto pode aparecer só por compartilhar um termo, sem relação de sentido.
- **BERT (CLS/MEAN)**: aproxima por significado; para `"vector database"` (inglês), o modelo multilíngue ainda associa aos textos em português sobre banco vetorial.
- **SBERT**: melhor resultado geral, treinado especificamente para essa tarefa.

## Visualização: t-SNE

- Projeta embeddings de alta dimensão para **2D preservando vizinhanças locais**.
- Ótimo para ver "clusters" visuais; **não preserva distâncias globais** (pode distorcer).
- Parâmetro importante: `perplexity` (ex.: 10, 15, 20) — afeta a granularidade dos agrupamentos.

## Avaliação: recall@k

- Ground truth manual: para cada query, define quais IDs são relevantes (`relevant_ids`).
- **recall@k**: para cada query, verifica se **pelo menos 1 item relevante** aparece nos `top-k`:

  ```
  recall@k = nº queries com ≥1 relevante nos top-k / nº total de queries
  ```

- Compara TF-IDF vs BERT vs SBERT para `k = 1, 3, 5`.

## Tópicos do exercício 5.5 (resumo das respostas)

1. **Abordagem mais semântica**: BERT — aproxima por significado, sem limitar às palavras exatas.
2. **Presa às palavras exatas**: TF-IDF — aparece por sobreposição de termos, sem conexão semântica.
3. **"vector database" (inglês)**: BERT multilíngue associa corretamente aos textos em português; TF-IDF se prende ao termo exato e entrega resultados sem relação.
