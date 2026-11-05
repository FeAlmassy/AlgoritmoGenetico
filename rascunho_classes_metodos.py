"""
RASCUNHO 2 — tudo dentro da classe, cada etapa virou um método.

Não faz parte do pacote, é só para ter uma noção.
Compare com rascunho_classes.py: lá as etapas eram funções de fora.
Aqui não precisa importar nada do pacote, a classe tem tudo.

Repare que, como X, y e a população ficam guardados no self,
a maioria dos métodos quase não recebe parâmetros.
"""

import numpy as np


class ClassificadorGenetico:
    """Classificador linear (binário ou ordinal) treinado por algoritmo genético."""

    def __init__(self, qtd_cromossomos: int = 6, qtd_genes_mutados: int = 1,
                 qtd_filhos: int = 3, qtd_substituicoes: int = 2,
                 geracoes: int = 100, fitness_alvo: float = 0.90,
                 thresholds: np.ndarray | None = None,
                 minimo: float = -1.0, maximo: float = 1.0,
                 seed: int | None = None, verbose: bool = True):

        if qtd_substituicoes > min(qtd_cromossomos, qtd_filhos):
            raise ValueError("qtd_substituicoes não pode passar de qtd_cromossomos nem de qtd_filhos")

        if minimo >= maximo:
            raise ValueError(f"minimo ({minimo}) deve ser menor que maximo ({maximo})")

        # hiperparâmetros
        self.qtd_cromossomos = qtd_cromossomos
        self.qtd_genes_mutados = qtd_genes_mutados
        self.qtd_filhos = qtd_filhos
        self.qtd_substituicoes = qtd_substituicoes
        self.geracoes = geracoes
        self.fitness_alvo = fitness_alvo
        self.thresholds = None if thresholds is None else np.asarray(thresholds)
        self.minimo = minimo
        self.maximo = maximo
        self.seed = seed
        self.verbose = verbose

        # estado do treino: só existe enquanto treinar() está rodando
        self.X = None
        self.y = None
        self.populacao = None

        # resultado
        self.cromossomo = None
        self.fitness = None
        self.acuracia = None
        self.geracao = None
        self.historico = None

    # ------------------------------------------------------------------
    # 1ª etapa: criação
    # ------------------------------------------------------------------
    def _criar_cromossomos(self) -> None:
        """Cria a população inicial e guarda em self.populacao."""
        qtd_genes = self.X.shape[1] + 1  # +1 por causa do bias
        self.populacao = np.random.uniform(self.minimo, self.maximo, size=(self.qtd_cromossomos, qtd_genes))

    # ------------------------------------------------------------------
    # 2ª etapa: fitness
    # ------------------------------------------------------------------
    def _calcular_fitness(self, cromossomos: np.ndarray) -> np.ndarray:
        """Escolhe o fitness binário ou ordinal olhando self.thresholds."""
        if self.thresholds is None:
            return self._fitness_binario(cromossomos)
        return self._fitness_ordinal(cromossomos)

    def _fitness_binario(self, cromossomos: np.ndarray) -> np.ndarray:
        """Recall da classe 1 vezes recall da classe 0, para cada cromossomo."""
        total_positivos = np.sum(self.y == 1)
        total_negativos = np.sum(self.y == 0)

        lista_fitness = []
        for linha in cromossomos:
            hipotese = self._classificar(linha, self.X)

            recall_pos = np.sum((hipotese == 1) & (self.y == 1)) / total_positivos
            recall_neg = np.sum((hipotese == 0) & (self.y == 0)) / total_negativos

            lista_fitness.append(recall_pos * recall_neg)

        return np.array(lista_fitness)

    def _fitness_ordinal(self, cromossomos: np.ndarray) -> np.ndarray:
        """Produto dos recalls de todas as K classes, para cada cromossomo."""
        K = len(self.thresholds) + 1
        totais = np.array([np.sum(self.y == k) for k in range(K)])

        lista_fitness = []
        for linha in cromossomos:
            hipotese = self._classificar(linha, self.X)

            fitness = 1.0
            for k in range(K):
                acertos_k = np.sum((hipotese == k) & (self.y == k))
                fitness *= acertos_k / totais[k]

            lista_fitness.append(fitness)

        return np.array(lista_fitness)

    @staticmethod
    def _fitness_percentual(vetor_fitnesses: np.ndarray) -> np.ndarray:
        """
        Transforma os fitnesses em chances que somam 1.
        É @staticmethod porque não usa nada do self, só o que recebe.
        """
        soma = np.sum(vetor_fitnesses)
        if soma == 0:
            return np.ones(len(vetor_fitnesses)) / len(vetor_fitnesses)
        return vetor_fitnesses / soma

    # ------------------------------------------------------------------
    # 3ª etapa: seleção
    # ------------------------------------------------------------------
    @staticmethod
    def _girar_roleta(percentual_fitnesses: np.ndarray) -> int:
        """Gira a roleta uma vez e devolve o índice sorteado."""
        sorteio = np.random.rand()
        soma = 0.0
        for indice, chance in enumerate(percentual_fitnesses):
            soma += chance
            if sorteio <= soma:
                return indice
        return len(percentual_fitnesses) - 1

    def _selecionar_pais_roleta(self, percentuais: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sorteia pai e mãe da self.populacao (método chamando outro método)."""
        indice_pai = self._girar_roleta(percentuais)
        indice_mae = self._girar_roleta(percentuais)
        return self.populacao[indice_pai], self.populacao[indice_mae]

    # ------------------------------------------------------------------
    # 4ª etapa: cruzamento
    # ------------------------------------------------------------------
    def _cruzar_um_ponto(self, pai: np.ndarray, mae: np.ndarray) -> np.ndarray:
        """Gera self.qtd_filhos filhos cortando os pais em um ponto aleatório."""
        filhos = []
        for _ in range(self.qtd_filhos):
            corte = np.random.randint(1, len(pai))
            filho = pai.copy()
            filho[corte:] = mae[corte:]
            filhos.append(filho)
        return np.array(filhos)

    # ------------------------------------------------------------------
    # 5ª etapa: mutação
    # ------------------------------------------------------------------
    def _mutar(self, filhos: np.ndarray) -> np.ndarray:
        """Troca self.qtd_genes_mutados genes de cada filho por valores aleatórios."""
        qtd_genes = len(filhos[0])
        filhos_mutados = filhos.astype(float)

        for filho in filhos_mutados:
            indices = np.random.choice(qtd_genes, size=self.qtd_genes_mutados, replace=False)
            for indice in indices:
                filho[indice] = -1 + 2 * np.random.rand()

        return filhos_mutados

    # ------------------------------------------------------------------
    # 6ª etapa: atualização
    # ------------------------------------------------------------------
    def _atualizar_populacao(self, fitnesses: np.ndarray, filhos: np.ndarray, fitnesses_filhos: np.ndarray) -> None:
        """
        Troca os piores da self.populacao pelos melhores filhos.
        Não retorna nada: altera self.populacao direto.
        """
        ordem_populacao = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])
        ordem_filhos = sorted(range(len(fitnesses_filhos)), key=lambda i: fitnesses_filhos[i])

        indices_piores = ordem_populacao[:self.qtd_substituicoes]
        indices_melhores_filhos = ordem_filhos[len(ordem_filhos) - self.qtd_substituicoes:]

        nova_populacao = self.populacao.copy()
        for i in range(self.qtd_substituicoes):
            nova_populacao[indices_piores[i]] = filhos[indices_melhores_filhos[i]]

        self.populacao = nova_populacao

    # ------------------------------------------------------------------
    # métodos públicos (sem _ na frente): é isso que o usuário chama
    # ------------------------------------------------------------------
    def treinar(self, X: np.ndarray, y: np.ndarray) -> "ClassificadorGenetico":
        """Roda o algoritmo genético e guarda o melhor cromossomo no objeto."""

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X tem {X.shape[0]} amostras, mas y tem {y.shape[0]} valores")

        if self.seed is not None:
            np.random.seed(self.seed)

        self.X = X
        self.y = y
        self.fitness = -np.inf
        self.acuracia = 0.0
        self.historico = {"melhor": [], "medio": [], "acuracia": []}

        self._criar_cromossomos()

        for geracao in range(self.geracoes):
            fitnesses = self._calcular_fitness(self.populacao)
            percentuais = self._fitness_percentual(fitnesses)

            idx_melhor = int(np.argmax(fitnesses))
            if fitnesses[idx_melhor] > self.fitness:
                self.fitness = float(fitnesses[idx_melhor])
                self.cromossomo = self.populacao[idx_melhor].copy()
                self.acuracia = float(np.mean(self.prever(X) == y))

            self.geracao = geracao + 1
            self.historico["melhor"].append(self.fitness)
            self.historico["medio"].append(float(fitnesses.mean()))
            self.historico["acuracia"].append(self.acuracia)

            if self.verbose:
                print(f"Geração {self.geracao:>4} | melhor fitness: {self.fitness:.4f} | acurácia: {self.acuracia:.2%}")

            if self.fitness >= self.fitness_alvo:
                if self.verbose:
                    print(f"\nFitness alvo {self.fitness_alvo} atingido na geração {self.geracao}.")
                break

            # compare com o rascunho 1: as chamadas ficaram bem mais curtas
            pai, mae = self._selecionar_pais_roleta(percentuais)
            filhos = self._cruzar_um_ponto(pai, mae)
            filhos = self._mutar(filhos)
            fitnesses_filhos = self._calcular_fitness(filhos)
            self._atualizar_populacao(fitnesses, filhos, fitnesses_filhos)

        # limpa o que só servia durante o treino
        self.X = None
        self.y = None
        self.populacao = None

        return self

    def _classificar(self, cromossomo: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Aplica um cromossomo em X. Usado pelo fitness e pelo prever."""
        bias = cromossomo[0]
        genes = cromossomo[1:]
        q = X @ genes + bias

        if self.thresholds is None:
            return np.where(q >= 0, 1, 0)
        return np.digitize(q, self.thresholds)

    def prever(self, X: np.ndarray) -> np.ndarray:
        """Classifica dados novos com o melhor cromossomo encontrado."""
        if self.cromossomo is None:
            raise ValueError("o modelo ainda não foi treinado, chame treinar() antes")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        return self._classificar(self.cromossomo, X)

    def __repr__(self) -> str:
        status = "não treinado" if self.cromossomo is None else f"fitness={self.fitness:.4f}"
        return f"ClassificadorGenetico(qtd_cromossomos={self.qtd_cromossomos}, geracoes={self.geracoes}, {status})"


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_excel("exemplos/exemplo_1_binario/dados_binario_1.xlsx")
    X = df.iloc[:, 1:-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()

    modelo = ClassificadorGenetico(qtd_cromossomos=10, geracoes=50, seed=42, verbose=False)
    print(modelo)

    modelo.treinar(X, y)
    print(modelo)
    print("acurácia:", modelo.acuracia)
