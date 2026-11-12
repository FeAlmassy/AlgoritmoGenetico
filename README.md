# Genetic Classifier

A Python library that trains a **linear classifier with a genetic algorithm**, written from scratch with NumPy.

Instead of using gradients (like logistic regression), the model **evolves** a population of solutions: the best ones reproduce, mutate and, generation after generation, the classification improves.

It works for:
- **Binary classification** (2 classes: 0 and 1)
- **Ordinal classification** (K ordered classes, e.g. bad < average < good)

---

## How it works

Each **chromosome** is a list of numbers: one **bias** and one **weight** per feature. With them the model computes a score for each sample:

```
score = bias + weight₁·x₁ + weight₂·x₂ + ... + weightₙ·xₙ
```

- **Binary:** score ≥ 0 → class 1, score < 0 → class 0
- **Ordinal:** the score is split into ranges by the **thresholds** (cut points), and each range is a class

### The fitness

The score of each chromosome (its fitness) is the **product of the recall of each class**:

```
fitness = recall(class 0) × recall(class 1) × ... × recall(class K-1)
```

If the model gets every sample of a class wrong, the fitness becomes 0. This forces the model to get **every** class right, not only the most common one, which is useful when the classes are imbalanced.

### The algorithm cycle

```
 1. create random population
          │
          ▼
 2. compute fitness ◄───────────────┐
          │                         │
          ▼                         │
 3. select two parents (roulette)   │
          │                         │
          ▼                         │
 4. crossover (one cut point)       │
          │                         │
          ▼                         │
 5. mutation                        │
          │                         │
          ▼                         │
 6. replace the worst with ─────────┘
    the best offspring
```

The cycle stops when it reaches the target fitness or runs out of generations.

| Step | What it does |
|---|---|
| **Creation** | Creates chromosomes with random genes between -1 and 1 |
| **Fitness** | Gives each chromosome a score from 0 to 1 |
| **Selection** | Draws two parents by roulette: the higher the fitness, the higher the chance |
| **Crossover** | The child takes the beginning of one parent and the end of the other, from a random cut point |
| **Mutation** | Replaces some genes of each child with new random values |
| **Update** | The worst of the population leave and the best offspring come in (elitism) |

---

## Installation

Requires **Python 3.10+**.

```bash
pip install genetic-classifier
```

Or straight from GitHub:

```bash
pip install git+https://github.com/FeAlmassy/Genetic-Classifier.git
```

### For development

```bash
git clone https://github.com/FeAlmassy/Genetic-Classifier.git
cd Genetic-Classifier
python -m venv venv_ag
source venv_ag/bin/activate        # on Windows: venv_ag\Scripts\activate
pip install -e .                   # editable mode: code changes apply right away
```

---

## Quick start

### Binary

```python
from genetic_classifier import train, predict, evaluate, read_excel, train_test_split

X, y = read_excel("examples/example_1_binary/binary_data_1.xlsx")
X_train, X_test, y_train, y_test = train_test_split(X, y, seed=0)

model = train(X_train, y_train, seed=0)    # learns from the training set only

y_pred = predict(model, X_test)            # classifies data the model has never seen
results = evaluate(y_test, y_pred)         # prints accuracy, fitness, recall and confusion matrix
```

> The default values of `train` already give good results in most cases. Only tune the parameters if you need to.

The functions of each step (`create_chromosomes`, `binary_fitness`, `select_parents_roulette`, `one_point_crossover`, `mutate`, `update_population`) can also be used on their own, imported from their modules (e.g. `from genetic_classifier.mutation import mutate`).

### Ordinal (several classes)

The classes must be numbers from `0` to `K-1`, ordered from lowest to highest. For K classes, pass K-1 thresholds in increasing order:

```python
import numpy as np

thresholds = np.array([-1.5, -0.5, 0.5, 1.5])   # 4 cuts -> 5 classes

model = train(X, y, thresholds=thresholds, n_generations=2000)
y_pred = predict(model, X_new)
```

### `train` parameters

| Parameter | Default | Description |
|---|---|---|
| `X` | — | array (n_samples, n_features) with the data |
| `y` | — | vector with the class of each sample |
| `n_chromosomes` | `30` | population size |
| `n_mutated_genes` | `2` | how many genes change in each child |
| `n_offspring` | `16` | how many children per generation |
| `n_replacements` | `None` | how many of the worst in the population are swapped for the best children (at most the smaller of `n_chromosomes` and `n_offspring`). `None` = half of `n_offspring` |
| `n_generations` | `1000` | maximum number of generations |
| `target_fitness` | `1.0` | stops when the best fitness reaches this value (`1.0` = runs every generation unless it gets everything right) |
| `thresholds` | `None` | `None` = binary; array with K-1 cuts = ordinal |
| `gene_min` | `-1.0` | smallest gene value in the initial population |
| `gene_max` | `1.0` | largest gene value in the initial population |
| `seed` | `None` | random seed for reproducibility (`None` = different every run) |
| `verbose` | `True` | prints the best fitness and accuracy of every generation |

`train` returns a dictionary with `chromosome` (the best one found), `fitness`, `accuracy` (fraction right on the training data), `generation` (where it stopped), `thresholds` and `history` (lists `best`, `mean` and `accuracy` with the evolution per generation).

### Tuning the parameters

The defaults work well in most cases. If you need to tune them, keep in mind that the parameters affect each other:

- **Renewal rate = `n_replacements` ÷ `n_chromosomes`:** fraction of the population replaced every generation (default: 8 ÷ 30 ≈ 27%). Too low makes evolution slow; close to 100% the whole population becomes children of the same couple and loses diversity. That is why **increasing only the population usually makes things worse**: increase the replacements too, or leave it as `None`.
- **`n_offspring` × `n_replacements`:** only the best children enter the population. With more children than replacements, the worst children are discarded.
- **`n_mutated_genes` × number of features:** what matters is the fraction of the chromosome that changes. With few features, use `1`; with many, `2` or `3`.
- **Cost ≈ `n_generations` × (`n_chromosomes` + `n_offspring`) × n_samples.**
- **Explore × refine:** a large population and more mutation search for new solutions; high renewal and little mutation improve the ones already found.

To find out whether more generations would help, look at the end of `model["history"]["best"]`: if the fitness is still going up, the model has not finished.

---

## Data format

The spreadsheet (`.xlsx` or `.csv`) must follow this layout:

| Lessons | X1 | X2 | ... | Xn | CLASS |
|---|---|---|---|---|---|
| Y001 | 1 | 0 | ... | 1 | 1 |
| Y002 | 0 | 1 | ... | 0 | 0 |

- **First column:** sample identifier (ignored)
- **Middle columns:** features
- **Last column:** class

To classify new data, which has no class column, use `has_labels=False`. In that case the function returns only `X`:

```python
X_new = read_excel("new_data.xlsx", has_labels=False)
y_pred = predict(model, X_new)
```

A ready-made template is in [`examples/template/template.xlsx`](https://github.com/FeAlmassy/Genetic-Classifier/blob/main/examples/template/template.xlsx).

---

## Examples and results

| Notebook | Dataset | Classes | Test accuracy* |
|---|---|---|---|
| [`binary_example.ipynb`](https://github.com/FeAlmassy/Genetic-Classifier/blob/main/examples/example_1_binary/binary_example.ipynb) | `binary_data_1.xlsx` (539 samples, 18 features) | 2 | **87%** |
| [`multiclass_example.ipynb`](https://github.com/FeAlmassy/Genetic-Classifier/blob/main/examples/example_2_multiclass/multiclass_example.ipynb) | `multiclass_data.xlsx` (261 samples, 30 features) | 5 | **79%** |
| [`mushroom_classification.ipynb`](https://github.com/FeAlmassy/Genetic-Classifier/blob/main/examples/mushroom_classification/mushroom_classification.ipynb) | `mushroom_data.xlsx` (8124 samples, 117 features) | 2 | **100%** |
| [`heart_disease_classification.ipynb`](https://github.com/FeAlmassy/Genetic-Classifier/blob/main/examples/heart_disease_classification/heart_disease_classification.ipynb) | `heart_data.xlsx` (917 samples, 21 features) | 2 | **85%** |
| [`diabetes_classification.ipynb`](https://github.com/FeAlmassy/Genetic-Classifier/blob/main/examples/diabetes_classification/diabetes_classification.ipynb) | `diabetes_data.xlsx` (40000 samples, 21 features) | 3 | **58%**† |

\* Trained on 80% of the data and tested on the remaining 20%, which the model never saw, always with seed 0.

† Prediabetes is very rare in this dataset, so the accuracy alone is misleading: the notebook shows the recall of each class.

The binary notebook shows **one generation step by step**, calling each function in the order of the algorithm. The multiclass one shows how to choose the order of the classes when it is not known.

---

## Project structure

```
Genetic-Classifier/
├── src/
│   └── genetic_classifier/
│       ├── __init__.py      # public API: train, predict, evaluate, confusion_matrix, read_excel, read_csv, train_test_split
│       ├── core.py          # main loop (train) and prediction (predict)
│       ├── evaluation.py    # model evaluation (evaluate) and confusion matrix
│       ├── creation.py      # 1. initial population
│       ├── fitness.py       # 2. binary and ordinal fitness, roulette proportions
│       ├── selection.py     # 3. roulette selection
│       ├── crossover.py     # 4. one-point crossover
│       ├── mutation.py      # 5. mutation
│       ├── update.py        # 6. replacing the worst with the best offspring
│       ├── reading.py       # reading .xlsx and .csv files
│       └── split.py         # train/test split (stratified)
├── docs/                    # how the algorithm works, with math and examples (Markdown and LaTeX)
├── examples/                # example notebooks and datasets
├── tests/                   # automated tests (coming soon)
├── pyproject.toml           # package configuration (name, version, dependencies)
├── LICENSE                  # MIT license
└── requirements.txt
```

Each step of the algorithm lives in its own module, so you can swap one piece (for example, try another selection method) without touching the others. Every function checks its parameters and gives clear error messages.

---

## Changelog

### 3.0.0
- **The whole library is now in English** and is published on PyPI as **`genetic-classifier`** (import `genetic_classifier`). The Portuguese version stays available as `algoritmo-genetico` 2.0.0.
- Every docstring follows the same format: what the function does, each parameter with its type, and what it returns with its type and a suggested variable name.
- The deprecated alias `rodar` was removed.

Migrating from `algoritmo-genetico`:

| 2.x (Portuguese) | 3.0 (English) |
|---|---|
| `treinar`, `prever`, `avaliar`, `matriz_confusao` | `train`, `predict`, `evaluate`, `confusion_matrix` |
| `ler_excel`, `ler_csv` (`tem_classe`) | `read_excel`, `read_csv` (`has_labels`) |
| `dividir_treino_teste` (`proporcao_treino`, `estratificar`) | `train_test_split` (`train_size`, `stratify`) |
| `qtd_cromossomos`, `qtd_genes_mutados`, `qtd_filhos`, `qtd_substituicoes` | `n_chromosomes`, `n_mutated_genes`, `n_offspring`, `n_replacements` |
| `geracoes`, `fitness_alvo`, `minimo`, `maximo` | `n_generations`, `target_fitness`, `gene_min`, `gene_max` |
| `y_real`, `y_previsto`, `nomes_classes` | `y_true`, `y_pred`, `class_names` |
| keys `cromossomo`, `acuracia`, `geracao`, `historico` (`melhor`, `medio`) | keys `chromosome`, `accuracy`, `generation`, `history` (`best`, `mean`) |
| key `matriz` of `avaliar` | key `confusion_matrix` of `evaluate` |
| `criar_cromossomos`, `fitness_binario`, `fitness_ordinal`, `fitness_percentual` | `create_chromosomes`, `binary_fitness`, `ordinal_fitness`, `fitness_proportions` |
| `selecionar_pais_roleta`, `cruzar_um_ponto`, `mutar`, `atualizar_populacao` | `select_parents_roulette`, `one_point_crossover`, `mutate`, `update_population` |

### 2.0.0
- New **`evaluate`** function (`avaliar`), which prints and returns accuracy, fitness, recall per class and the confusion matrix. The matrix can also be obtained on its own with **`confusion_matrix`**.
- `read_excel` and `read_csv` got the **`has_labels`** parameter: with `has_labels=False` they read new data without the class column.
- New checks: empty cells, text in the features and text classes in ordinal mode now raise clear error messages instead of silently giving wrong results.
- Training no longer creates offspring in the last generation, since they would never be compared with the best chromosome.

### 1.1.0
- `rodar` was renamed to **`treinar`** (now `train`), pairing with `prever` (now `predict`).
- **New default values**, which give much better results without tuning: 30 chromosomes, 2 mutated genes, 16 offspring, replacements = half of the offspring, 1000 generations and target fitness 1.0.
- Training converts `X` to float internally, which makes it up to ~6x faster when the data is integer (as read by `read_csv`).

---

## License

Distributed under the [MIT license](https://github.com/FeAlmassy/Genetic-Classifier/blob/main/LICENSE): you can use, modify and redistribute it freely, as long as you keep the copyright notice.

---

## Authors

- **Fellipe Almassy** — [GitHub](https://github.com/FeAlmassy)
- **Matheus Gomes** — [GitHub](https://github.com/matheos-hash)
