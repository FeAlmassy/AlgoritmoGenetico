Estou aprendendo programação orientada a objetos (classes) usando a minha própria biblioteca de algoritmo genético como exemplo. Quero continuar a aula de onde parei.

## Contexto do projeto

- Biblioteca: ~/Projects/AlgoritmoGenetico, branch `3.0.0-EN-Version` (ainda sem commit). Já foi toda traduzida para inglês e vai ser relançada no PyPI como `genetic-classifier` (import `genetic_classifier`), versão 3.0.0.
- Hoje a API é por funções: `model = train(X, y, ...)` devolve um dict (`chromosome`, `fitness`, `accuracy`, `generation`, `thresholds`, `history`) e `predict(model, X)` recebe esse dict de volta.
- As etapas do AG são funções puras em módulos separados: `create_chromosomes` (creation.py), `binary_fitness`/`ordinal_fitness`/`fitness_proportions` (fitness.py), `select_parents_roulette` (selection.py), `one_point_crossover` (crossover.py), `mutate` (mutation.py), `update_population` (update.py). O loop fica em core.py.
- Existe um rascunho `classes_draft.py` na raiz do projeto (vai ser apagado depois) com `_GeneticLinearModel` (classe base com o loop do AG), `GeneticClassifier` e `GeneticRegressor` (fitness = 1/(1+MSE), padroniza X e y). Segue as convenções do scikit-learn sem depender dele: `__init__` só guarda os parâmetros, `fit` valida/treina e retorna `self`, o que foi aprendido termina em `_` (`coef_`, `intercept_`, `fitness_`, `history_`), `predict` e `score`. Decidi NÃO usar scikit-learn por enquanto; vou adicionar a herança de `BaseEstimator` mais pra frente, quando aprender.

## Como você está me ensinando (mantenha esse formato)

- **Uma etapa por mensagem.** Nunca mande várias etapas de uma vez. No fim de cada etapa, diga qual é a próxima e pergunte se quero seguir. Espere eu responder.
- Explicação em português, código em inglês.
- Sempre usar o MEU programa como exemplo (`GeneticClassifier`, `n_generations`, `train`, `predict`, `model["chromosome"]`...), não exemplos genéricos de cachorro/carro.
- Mostrar sempre o **antes (funções + dict) x depois (classe)** lado a lado.
- Trechos de código curtos, com setas/diagramas em texto quando ajudar (ex: mostrando quem vira o `self`).
- Analogia usada: classe = planta da casa, objeto = casa construída; várias casas da mesma planta são independentes.
- Se eu perguntar algo no meio (ex: "o que significa clf?"), responda só a dúvida e pergunte se quero seguir.

## O que já foi ensinado

1. **Classe x objeto:** `class GeneticClassifier: pass` e `clf = GeneticClassifier()`. Hoje dados (o dict) e comportamento (`train`/`predict`) vivem separados; a classe junta os dois. Objetos independentes (`clf_a`, `clf_b`).
   - Dúvida respondida: `clf` é só abreviação de classifier, convenção da comunidade (`clf`, `reg`, `model`).
2. **Atributos:** `clf.n_generations = 500`, acesso com ponto; cada objeto tem os seus. Comparação com dict (`model["n_generations"]`): até aqui só muda a sintaxe, a diferença real é que o objeto pode ter métodos. Problema de preencher na mão → motiva o `__init__`.
3. **`__init__` e `self`:** chamado automaticamente ao criar o objeto; os 4 passos (cria vazio → chama `__init__` com o objeto como `self` → `self.x = ...` grava → devolve). `self.n_generations = n_generations` (atributo x parâmetro). Fora da classe é `clf`, dentro é `self`. Os parâmetros do `__init__` são os mesmos do `train` de hoje; `X` e `y` NÃO entram no `__init__` (vão no `fit`). Regra do sklearn: `__init__` só guarda.

## Onde parei

Comece pela **Etapa 4: métodos** (eu ainda não li essa). Plano das próximas etapas:

4. **Métodos:** função dentro da classe que recebe `self` e enxerga os atributos (ex: `describe()`); `clf.describe()` é o mesmo que `GeneticClassifier.describe(clf)`; ligação com `predict(model, X)` → `clf.predict(X)` (o `model` já fazia o papel do `self`); métodos também podem alterar atributos (`self.x = ...`), preparando o `fit`.
5. **`fit`:** o corpo do `train` vira um método; em vez de `return {...}`, grava `self.coef_`, `self.intercept_`, `self.fitness_` e faz `return self`. Convenção do `_` no final (aprendido x configurado). Ciclo de vida: nasce só com configuração → depois do `fit` ganha os pesos.
6. **`predict`:** usa `self.coef_`/`self.intercept_`; comparar com as 6 checagens do `predict` atual sobre o dict (somem porque o próprio `fit` criou os atributos); `_check_is_fitted`.
7. **Herança:** classificação e regressão só diferem no fitness e em como o score vira previsão; classe base `_GeneticLinearModel` com o loop e dois "buracos" (`_fitness`, `_scores_to_prediction`) que cada subclasse preenche (padrão template method); `super().__init__`.
8. **Fechamento:** ler o `classes_draft.py` inteiro com tudo que foi aprendido; quando usar classe x função (as etapas do AG continuam funções puras); cuidados (estado mutável: `fit` duas vezes sobrescreve; não exagerar na herança); como seria plugar o scikit-learn depois.
