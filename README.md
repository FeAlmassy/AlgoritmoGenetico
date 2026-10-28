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
pip install algoritmo-genetico
```

Ou direto do GitHub:

```bash
pip install git+https://github.com/FeAlmassy/AlgoritmoGenetico.git
```

### Para desenvolver

```bash
git clone https://github.com/FeAlmassy/AlgoritmoGenetico.git
cd AlgoritmoGenetico
python -m venv venv_ag
source venv_ag/bin/activate        # no Windows: venv_ag\Scripts\activate
pip install -e .                   # modo editável: mudanças no código valem na hora
```

---

## Uso rápido

### Binário

```python
from algoritmo_genetico import treinar, prever, ler_excel, dividir_treino_teste

X, y = ler_excel("exemplos/exemplo_1_binario/dados_binario_1.xlsx")
X_treino, X_teste, y_treino, y_teste = dividir_treino_teste(X, y, seed=0)

modelo = treinar(X_treino, y_treino, seed=0)    # aprende só com o treino

previsoes = prever(modelo, X_teste)             # classifica dados que o modelo nunca viu
```

> Os valores padrão do `treinar` já dão um bom resultado na maioria dos casos. Ajuste os parâmetros só se precisar.

As funções de cada etapa (`criar_cromossomos`, `fitness_binario`, `selecionar_pais_roleta`, `cruzar_um_ponto`, `mutar`, `atualizar_populacao`) também podem ser usadas separadamente, importando dos seus módulos (ex: `from algoritmo_genetico.mutacao import mutar`).

### Ordinal (várias classes)

As classes precisam ser números de `0` a `K-1`, na ordem da menor para a maior. Para K classes, passe K-1 thresholds em ordem crescente:

```python
import numpy as np

thresholds = np.array([-1.5, -0.5, 0.5, 1.5])   # 4 cortes -> 5 classes

modelo = treinar(X, y, thresholds=thresholds, geracoes=2000)
previsoes = prever(modelo, X_novo)
```

### Parâmetros do `treinar`

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `X` | — | array (amostras, features) com os dados |
| `y` | — | vetor com a classe de cada amostra |
| `qtd_cromossomos` | `30` | tamanho da população |
| `qtd_genes_mutados` | `2` | quantos genes mudam em cada filho |
| `qtd_filhos` | `16` | quantos filhos por geração |
| `qtd_substituicoes` | `None` | quantos piores da população são trocados pelos melhores filhos (no máximo o menor entre `qtd_cromossomos` e `qtd_filhos`). `None` = metade de `qtd_filhos` |
| `geracoes` | `1000` | número máximo de gerações |
| `fitness_alvo` | `1.0` | para quando o melhor fitness chega nesse valor (`1.0` = roda todas as gerações, a não ser que acerte tudo) |
| `thresholds` | `None` | `None` = binário; array com K-1 cortes = ordinal |
| `minimo` | `-1.0` | menor valor dos genes na população inicial |
| `maximo` | `1.0` | maior valor dos genes na população inicial |
| `seed` | `None` | semente para reprodutibilidade (`None` = aleatório a cada rodada) |
| `verbose` | `True` | mostra o melhor fitness e a acurácia de cada geração |

O `treinar` devolve um dicionário com `cromossomo` (o melhor encontrado), `fitness`, `acuracia` (fração de acertos nos dados de treino), `geracao` (em que geração parou), `thresholds` e `historico` (listas `melhor`, `medio` e `acuracia` com a evolução por geração).

> **`rodar` → `treinar`:** a partir da versão 1.1.0, a função `rodar` se chama `treinar`. O nome antigo ainda funciona, mas mostra um aviso (`DeprecationWarning`) e será removido numa versão futura.

### Ajustando os parâmetros

Os padrões funcionam bem na maioria dos casos. Se precisar ajustar, os parâmetros se afetam entre si:

- **Taxa de renovação = `qtd_substituicoes` ÷ `qtd_cromossomos`:** fração da população trocada a cada geração (padrão: 8 ÷ 30 ≈ 27%). Muito baixa deixa a evolução lenta; perto de 100% a população vira toda filhos do mesmo casal e perde diversidade. Por isso, **aumentar só a população costuma piorar**: aumente as substituições junto, ou deixe `None`.
- **`qtd_filhos` × `qtd_substituicoes`:** só os melhores filhos entram na população. Com mais filhos do que substituições, os piores filhos são descartados.
- **`qtd_genes_mutados` × número de features:** o que importa é a fração do cromossomo que muda. Com poucas features, use `1`; com muitas, `2` ou `3`.
- **Custo ≈ `geracoes` × (`qtd_cromossomos` + `qtd_filhos`) × amostras.**
- **Explorar × refinar:** população grande e mais mutação procuram soluções novas; renovação alta e pouca mutação melhoram as que já existem.

Para saber se vale dar mais gerações, olhe o final de `modelo["historico"]["melhor"]`: se o fitness ainda está subindo, o modelo não terminou.

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

Um modelo pronto está em [`exemplos/base_de_dados_para_exemplos/modelo_base.xlsx`](https://github.com/FeAlmassy/AlgoritmoGenetico/blob/main/exemplos/base_de_dados_para_exemplos/modelo_base.xlsx).

---

## Exemplos e resultados

| Notebook | Dataset | Classes | Acurácia no teste* |
|---|---|---|---|
| [`exemplo_binario.ipynb`](https://github.com/FeAlmassy/AlgoritmoGenetico/blob/main/exemplos/exemplo_1_binario/exemplo_binario.ipynb) | `dados_binario_1.xlsx` (539 amostras, 18 features) | 2 | **87%** |
| [`exemplo_multiclasse.ipynb`](https://github.com/FeAlmassy/AlgoritmoGenetico/blob/main/exemplos/exemplo_2_multiclasse/exemplo_multiclasse.ipynb) | `dados_multiclasse.xlsx` (261 amostras, 30 features) | 5 | **79%** |

\* Treino com 80% dos dados e teste com os 20% restantes, que o modelo nunca viu. Com `np.random.seed(0)`.

O notebook binário mostra **uma geração passo a passo**, chamando cada função na ordem do algoritmo. O multiclasse mostra como escolher a ordem das classes quando ela não é conhecida.

---

## Estrutura do projeto

```
AlgoritmoGenetico/
├── src/
│   └── algoritmo_genetico/
│       ├── __init__.py      # API pública: treinar, prever, ler_excel, ler_csv, dividir_treino_teste
│       ├── nucleo.py        # loop principal (treinar) e previsão (prever)
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
├── pyproject.toml           # configuração do pacote (nome, versão, dependências)
├── LICENSE                  # licença MIT
└── requirements.txt
```

Cada etapa do algoritmo fica em um módulo próprio, então dá para trocar uma peça (por exemplo, testar outra seleção) sem mexer nas outras. Todas as funções verificam os parâmetros recebidos e dão mensagens de erro claras.

---

## Novidades

### 1.1.0
- `rodar` agora se chama **`treinar`**, formando par com `prever`. O nome antigo continua funcionando, com aviso de depreciação.
- **Novos valores padrão**, que dão resultados bem melhores sem precisar ajustar nada: `qtd_cromossomos=30`, `qtd_genes_mutados=2`, `qtd_filhos=16`, `qtd_substituicoes=None` (metade dos filhos), `geracoes=1000` e `fitness_alvo=1.0`.
- `treinar` converte `X` para float internamente, o que deixa o treino até ~6x mais rápido quando os dados são inteiros (como os lidos pelo `ler_csv`).

---

## Próximos passos

- [x] Empacotar com `pyproject.toml` e publicar no PyPI
- [ ] Testes automáticos com `pytest`
- [ ] Classe no estilo scikit-learn (`treinar` / `prever`)
- [ ] Salvar e carregar modelos treinados
- [ ] Aceitar classes em texto (ex: `"bom"`, `"ruim"`) sem conversão manual

---

## Licença

Distribuído sob a licença [MIT](https://github.com/FeAlmassy/AlgoritmoGenetico/blob/main/LICENSE): pode usar, modificar e redistribuir livremente, desde que mantenha o aviso de copyright.

---

## Autores

- **Fellipe Almassy** — [GitHub](https://github.com/FeAlmassy)
- **Matheus Gomes** — [GitHub](https://github.com/matheos-hash)
