# Algoritmo Genético para Classificação

Biblioteca em Python que treina um **classificador linear usando algoritmo genético**, escrita do zero com NumPy.

Em vez de usar gradiente (como a regressão logística), o modelo **evolui** uma população de soluções: as melhores se reproduzem, sofrem mutações e, geração após geração, a classificação melhora.

Funciona para:
- **Classificação binária** (2 classes: 0 e 1)
- **Classificação ordinal** (K classes com ordem, ex: ruim < médio < bom)

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
```

Se o modelo errar todas as amostras de uma classe, o fitness vira 0. Isso obriga o modelo a acertar **todas** as classes, e não só a mais comum, o que é útil quando as classes são desbalanceadas.

### O ciclo do algoritmo

```
 1. criar população aleatória
          │
          ▼
 2. calcular fitness ◄──────────────┐
          │                         │
          ▼                         │
 3. selecionar pai e mãe (roleta)   │
          │                         │
          ▼                         │
 4. cruzamento (corte em 1 ponto)   │
          │                         │
          ▼                         │
 5. mutação                         │
          │                         │
          ▼                         │
 6. trocar os piores pelos ─────────┘
    melhores filhos
```

O ciclo para quando atinge o fitness alvo ou acaba o número de gerações.

| Etapa | O que faz |
|---|---|
| **Criação** | Gera cromossomos com genes aleatórios entre -1 e 1 |
| **Fitness** | Dá uma nota de 0 a 1 para cada cromossomo |
| **Seleção** | Sorteia pai e mãe por roleta: quem tem fitness maior tem mais chance |
| **Cruzamento** | O filho pega o começo do pai e o final da mãe, a partir de um ponto de corte aleatório |
| **Mutação** | Troca alguns genes de cada filho por valores aleatórios novos |
| **Atualização** | Os piores da população saem e os melhores filhos entram (elitismo) |

---

## Instalação

Requer **Python 3.10+**.

```bash
git clone https://github.com/FeAlmassy/AlgoritmoGenetico.git
cd AlgoritmoGenetico
python -m venv venv_ag
source venv_ag/bin/activate        # no Windows: venv_ag\Scripts\activate
pip install -r requirements.txt
```

> Ainda não está no PyPI. Por enquanto, adicione a pasta `src` ao caminho do Python para importar a biblioteca (veja o exemplo abaixo).

---

## Uso rápido

### Binário

```python
import sys
sys.path.insert(0, "src")

from algoritmo_genetico import rodar, prever
from algoritmo_genetico.leitura import ler_excel

X, y = ler_excel("exemplos/exemplo_1_binario/dados_binario_1.xlsx")

modelo = rodar(X, y, geracoes=500, fitness_alvo=0.90)

previsoes = prever(modelo, X)
```

### Ordinal (várias classes)

As classes precisam ser números de `0` a `K-1`, na ordem da menor para a maior. Para K classes, passe K-1 thresholds em ordem crescente:

```python
import numpy as np

thresholds = np.array([-1.5, -0.5, 0.5, 1.5])   # 4 cortes -> 5 classes

modelo = rodar(X, y, thresholds=thresholds, geracoes=2000, fitness_alvo=1.0)
previsoes = prever(modelo, X_novo)
```

### Parâmetros do `rodar`

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `X` | — | array (amostras, features) com os dados |
| `y` | — | vetor com a classe de cada amostra |
| `qtd_cromossomos` | `6` | tamanho da população |
| `qtd_genes_mutados` | `1` | quantos genes mudam em cada filho |
| `qtd_filhos` | `3` | quantos filhos por geração |
| `qtd_substituicoes` | `2` | quantos piores da população são trocados pelos melhores filhos (no máximo o menor entre `qtd_cromossomos` e `qtd_filhos`) |
| `geracoes` | `100` | número máximo de gerações |
| `fitness_alvo` | `0.90` | para quando o melhor fitness chega nesse valor |
| `thresholds` | `None` | `None` = binário; array com K-1 cortes = ordinal |
| `minimo` | `-1.0` | menor valor dos genes na população inicial |
| `maximo` | `1.0` | maior valor dos genes na população inicial |
| `seed` | `None` | semente para reprodutibilidade (`None` = aleatório a cada rodada) |
| `verbose` | `True` | mostra o melhor fitness de cada geração |

O `rodar` devolve um dicionário com `cromossomo` (o melhor encontrado), `fitness`, `geracao` (em que geração parou), `thresholds` e `historico` (listas `melhor` e `medio` com a evolução do fitness por geração).

---

## Formato dos dados

A planilha (`.xlsx` ou `.csv`) deve seguir este layout:

| Lições | X1 | X2 | ... | Xn | CLASS |
|---|---|---|---|---|---|
| Y001 | 1 | 0 | ... | 1 | 1 |
| Y002 | 0 | 1 | ... | 0 | 0 |

- **Primeira coluna:** identificador da amostra (é ignorada)
- **Colunas do meio:** features
- **Última coluna:** classe

Um modelo pronto está em [`exemplos/base_de_dados_para_exemplos/modelo_base.xlsx`](exemplos/base_de_dados_para_exemplos/modelo_base.xlsx).

---

## Exemplos e resultados

| Notebook | Dataset | Classes | Acurácia no teste* |
|---|---|---|---|
| [`exemplo_binario.ipynb`](exemplos/exemplo_1_binario/exemplo_binario.ipynb) | `dados_binario_1.xlsx` (539 amostras, 18 features) | 2 | **87%** |
| [`exemplo_multiclasse.ipynb`](exemplos/exemplo_2_multiclasse/exemplo_multiclasse.ipynb) | `dados_multiclasse.xlsx` (261 amostras, 30 features) | 5 | **79%** |

\* Treino com 80% dos dados e teste com os 20% restantes, que o modelo nunca viu. Com `np.random.seed(0)`.

O notebook binário mostra **uma geração passo a passo**, chamando cada função na ordem do algoritmo. O multiclasse mostra como escolher a ordem das classes quando ela não é conhecida.

---

## Estrutura do projeto

```
AlgoritmoGenetico/
├── src/
│   └── algoritmo_genetico/
│       ├── __init__.py      # API pública: rodar, prever
│       ├── nucleo.py        # loop principal (rodar) e previsão (prever)
│       ├── criacao.py       # 1. população inicial
│       ├── fitness.py       # 2. fitness binário, ordinal e percentual
│       ├── selecao.py       # 3. seleção por roleta
│       ├── cruzamento.py    # 4. cruzamento de 1 ponto
│       ├── mutacao.py       # 5. mutação
│       ├── atualizacao.py   # 6. troca dos piores pelos melhores filhos
│       ├── leitura.py       # leitura de .xlsx e .csv
│       └── divisao.py       # divisão em treino e teste (estratificada)
├── exemplos/                # notebooks e datasets de exemplo
├── testes/                  # testes automáticos
└── requirements.txt
```

Cada etapa do algoritmo fica em um módulo próprio, então dá para trocar uma peça (por exemplo, testar outra seleção) sem mexer nas outras. Todas as funções verificam os parâmetros recebidos e dão mensagens de erro claras.

---

## Próximos passos

- [ ] Empacotar com `pyproject.toml` e publicar no PyPI
- [ ] Testes automáticos com `pytest`
- [ ] Classe no estilo scikit-learn (`treinar` / `prever`)
- [ ] Salvar e carregar modelos treinados
- [ ] Aceitar classes em texto (ex: `"bom"`, `"ruim"`) sem conversão manual

---

## Autor

**Fellipe Almassy** — [GitHub](https://github.com/FeAlmassy)
