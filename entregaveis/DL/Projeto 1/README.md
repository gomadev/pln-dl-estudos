Dicionario de dados

https://www.kaggle.com/datasets/mikhail1681/walmart-sales


Store: Store number
Date: Sales week start date
Weekly_Sales: Sales
Holiday_Flag: Mark on the presence or absence of a holiday
Temperature: Air temperature in the region
Fuel_Price: Fuel cost in the region
CPI: Consumer price index
Unemployment: Unemployment rate


## Data Augmentation (pipeline encadeado)

Pipeline encadeado como no material do professor (`materiais/DL/Projeto 1 - Regressão Linear.ipynb`): baseline -> SMOTE -> Mixup -> Ruído Gaussiano, onde cada técnica usa a saída da anterior. Tudo está no notebook `Projeto 1 - Regressão Linear (Walmart Sales).ipynb`.

Etapas:
- Alvo discretizado em 10 quantis (0-9, escala do vinho) para agrupamento e máscaras
- SMOTE(k_neighbors=3) nas classes de borda (<5 ou >7), classes 5-7 mantidas; amostras sintéticas recebem a média do Weekly_Sales da classe
- Mixup(alpha=0.2, n_samples=8) por classe, só amostras sintéticas, y interpolado
- Ruído gaussiano (1 cópia ruidosa por amostra) por classe, sigma = 10% do desvio padrão por feature, ruído somente nas colunas contínuas

Tamanhos: 6.435 -> 6.438 (SMOTE quase neutro, classes já balanceadas pelos quantis) -> 51.504 (mixup 8x) -> 103.008 (gaussiano 2x) ~ 100 mil.

Qualidade:
- Sem NaN/Inf e sem Weekly_Sales negativos em todas as bases
- Holiday_Flag sempre binária (arredondada para 0/1 nas saídas de SMOTE e mixup)
- Sem extrapolação (mixup e SMOTE interpolam dentro do domínio)
- Resultado empírico da execução: com o treino corrigido (padronização de y + mini-batches), a augmentação superou o baseline — MAE: 435,6K (Original) vs 418,8K (SMOTE), 384,6K (Mixup), 389,4K (Gaussiano). O mixup foi o melhor, ~12% abaixo do baseline.

Arquivos:
- data/wsales_smote.csv (6.438 linhas)
- data/wsales_mixup.csv (51.504 linhas)
- data/wsales_gaussian_noise.csv (103.008 linhas)
