# Segmentação de Clientes com Clustering

Mini projeto de **IA não supervisionada (ML - Clustering)**. A ideia é agrupar os clientes de um varejista em segmentos que compram de forma parecida, sem usar rótulos, para que a empresa possa direcionar produtos e campanhas para cada grupo.

**Integrantes:**
- Guilherme Schnekenberg Teixeira (RA00336189)
- Luís Augusto Coelho de Souza (RA00331675)


**Dataset:** Customer Personality Analysis (Kaggle) (https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)

---

## Estrutura do repositório

```
├── customer_segmentation_clustering.ipynb   # notebook com EDA + clustering (já executado)
├── marketing_campaign.csv                   # dataset (separado por TAB)
├── requirements.txt
└── README.md
```

## Etapas do notebook

1. **Visão geral dos dados:** tipos das colunas, estatísticas e colunas constantes.
2. **Dados faltantes com missingno:** só `Income` tem nulos (24 linhas, ≈1%), sem nenhum padrão. Essas linhas foram removidas.
3. **Limpeza e feature engineering:** criação de `Age`, `Spent`, `Children`, `Family_Size`, `Is_Parent`, `Living_With`, `Customer_For`, `NumPurchases` e `Total_Promos`, e simplificação de `Education`.
4. **EDA com matplotlib/seaborn:**
   - histogramas com média e mediana, e a assimetria de cada variável;
   - boxplots e remoção de outliers (idade > 90 e renda > 600k);
   - variáveis categóricas;
   - gasto por produto e compras por canal;
   - renda × gasto, pairplot e mapa de correlação.
5. **Pré-processamento:** LabelEncoder, StandardScaler e **PCA com 3 componentes** (56,7% da variância). As colunas de campanhas ficam fora do modelo e são usadas só para validar os grupos.
6. **Escolha do k:** método do cotovelo e silhouette, resultando em **k = 4**.
7. **Comparação de modelos:** K-Means, Agglomerative (Ward), Gaussian Mixture, Affinity Propagation e DBSCAN.
8. **Perfil dos clusters:** distribuições, produtos, canais, aceite de campanhas, família e personas.

## Resultados

### Comparação dos modelos

| Modelo | Clusters | Silhouette ↑ | Calinski-Harabasz ↑ | Davies-Bouldin ↓ |
|---|---|---|---|---|
| **K-Means** | 4 | **0.372** | **2089** | 0.993 |
| Agglomerative (Ward) | 4 | 0.355 | 1978 | 1.023 |
| Gaussian Mixture | 4 | 0.314 | 1605 | 0.981 |
| Affinity Propagation (pref = -1000) | 5 | 0.344 | 1865 | 1.110 |
| DBSCAN (eps = 0.8) | 3 + 181 ruído | 0.172 | 848 | 0.911* |

\* Calculado só sobre os pontos que não foram marcados como ruído.

**Modelo escolhido: K-Means com k = 4.** Ele teve as melhores métricas gerais e é fácil de explicar. O Agglomerative chegou a grupos muito parecidos, o que reforça que a estrutura de 4 grupos existe de fato nos dados. Já o DBSCAN não se encaixa bem, porque os dados formam uma nuvem contínua, sem regiões densas separadas.

### Personas

| Cluster | % base | Persona | Renda média | Gasto médio | Destaque |
|---|---|---|---|---|---|
| 0 | ≈23% | **Premium sem filhos** | ≈75k | ≈1.370 | compra vinho e carne, em loja e catálogo; é quem mais aceita campanhas |
| 2 | ≈26% | **Pais estabelecidos, caçadores de ofertas** | ≈62k | ≈890 | filhos adolescentes; usa todos os canais; é quem mais compra em promoção |
| 3 | ≈24% | **Famílias grandes, orçamento apertado** | ≈43k | ≈150 | maior família (≈3,5 pessoas); raramente aceita campanhas |
| 1 | ≈27% | **Jovens de baixa renda** | ≈30k | ≈100 | grupo mais jovem; visita muito o site, mas compra pouco |

### Recomendações

- **Premium:** produtos premium e catálogo; é o melhor público para novas campanhas.
- **Pais estabelecidos:** cupons, combos para família e comunicação pelo site.
- **Famílias grandes:** itens básicos em volume e descontos progressivos.
- **Jovens:** melhorar a conversão do site (frete, desconto na primeira compra, preços de entrada).

## O que foi feito

- limpeza de dados (remoção de nulos e cortes de outliers);
- pipeline LabelEncoder - StandardScaler - PCA com 3 componentes;
- perfil por renda x gasto e por campanhas;
- análise de nulos com missingno;
- EDA de distribuições em matplotlib (histogramas, assimetria, boxplots, categóricas, produtos e canais);
- silhouette junto com o cotovelo na escolha do k;
- comparação de 5 algoritmos com 3 métricas internas, incluindo a busca dos parâmetros do Affinity Propagation e do DBSCAN;
- escolha do K-Means como modelo final;
- personas e recomendações.

## Como executar

**Google Colab:** abra o notebook e rode tudo. O dataset é lido direto do repositório GitHub:

```python
url = "https://raw.github.com/schnekenberg/eda-customer-segmentation-clustering/main/marketing_campaign.csv"
df_raw = pd.read_csv(url, sep="\t")
```

Se der erro de módulo, rode `!pip install missingno`.

**Localmente:**

```bash
pip install -r requirements.txt
jupyter notebook customer_segmentation_clustering.ipynb
```

## Tecnologias

Python · pandas · NumPy · matplotlib · seaborn · missingno · scikit-learn
