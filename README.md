# Algoritmo Genético para Classificação

Biblioteca em Python que treina um **classificador linear usando algoritmo genético**, escrita do zero com NumPy.

Em vez de usar gradiente (como a regressão logística), o modelo **evolui** uma população de soluções: as melhores se reproduzem, sofrem mutações e, geração após geração, a classificação melhora.

Funciona para:
- **Classificação binária** (2 classes: 0 e 1)
- **Classificação não binária** (K classes com ordem, ex: ruim < médio < bom)

---

## Como funciona

Cada **cromossomo** é uma lista de números: um **bias** e um **peso** para cada feature. Com eles o modelo calcula uma nota para cada amostra:

```
nota = bias + peso₁·x₁ + peso₂·x₂ + ... + pesoₙ·xₙ
```

- **Binário:** nota ≥ 0 → classe 1, nota < 0 → classe 0
- **Ordinal:** a nota é dividida em faixas pelos **thresholds** (pontos de corte), e cada faixa é uma classe

### O fitness

A nota de cada cromossomo (fitness) é o **produto do recall de cada classe**:

```
fitness = recall(classe 0) × recall(classe 1) × ... × recall(classe K-1)
