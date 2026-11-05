"""
RASCUNHO — como a biblioteca ficaria usando classes.

Não faz parte do pacote, é só para ter uma noção.
As etapas (criação, seleção, cruzamento, mutação, atualização, fitness)
continuam sendo as funções que já existem; só o modelo virou classe.
"""

import numpy as np

from algoritmo_genetico.atualizacao import atualizar_populacao
from algoritmo_genetico.criacao import criar_cromossomos
from algoritmo_genetico.cruzamento import cruzar_um_ponto
from algoritmo_genetico.fitness import fitness_binario, fitness_ordinal, fitness_percentual
from algoritmo_genetico.mutacao import mutar
from algoritmo_genetico.selecao import selecionar_pais_roleta


class ClassificadorGenetico:
    """
    Classificador linear treinado por algoritmo genético.

    Os hiperparâmetros ficam guardados no objeto (self) quando ele é criado.
    Depois de treinar(), o objeto também guarda o resultado (cromossomo,
    fitness, histórico...), então prever() não precisa receber um dict.
    """

    def __init__(self, qtd_cromossomos: int = 6, qtd_genes_mutados: int = 1,
                 qtd_filhos: int = 3, qtd_substituicoes: int = 2,
                 geracoes: int = 100, fitness_alvo: float = 0.90,
                 thresholds: np.ndarray | None = None,
                 minimo: float = -1.0, maximo: float = 1.0,
                 seed: int | None = None, verbose: bool = True):

        # a validação dos hiperparâmetros mora aqui, roda uma vez só
        # (no pacote de verdade entrariam todas as checagens do rodar())
        if qtd_substituicoes > min(qtd_cromossomos, qtd_filhos):
            raise ValueError("qtd_substituicoes não pode passar de qtd_cromossomos nem de qtd_filhos")

        if minimo >= maximo:
            raise ValueError(f"minimo ({minimo}) deve ser menor que maximo ({maximo})")

        # hiperparâmetros: o que o usuário escolhe
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

        # resultado: começa vazio, é preenchido pelo treinar()
        self.cromossomo = None
        self.fitness = None
        self.acuracia = None
        self.geracao = None
        self.historico = None

    def _calcular_fitness(self, cromossomos: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Escolhe o fitness binário ou ordinal olhando os thresholds guardados no objeto."""
        if self.thresholds is None:
            return fitness_binario(cromossomos, X, y)
        return fitness_ordinal(cromossomos, X, y, self.thresholds)

    def treinar(self, X: np.ndarray, y: np.ndarray) -> "ClassificadorGenetico":
        """
        Roda o algoritmo genético e guarda o melhor cromossomo no próprio objeto.

        Parâmetros:
            X -> array (amostras, features) com os dados
            y -> vetor com a classe certa de cada amostra

        Retorna:
            o próprio objeto (dá para fazer modelo.treinar(X, y).prever(X))
        """

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X tem {X.shape[0]} amostras, mas y tem {y.shape[0]} valores")

        if self.seed is not None:
            np.random.seed(self.seed)

        qtd_genes = X.shape[1] + 1  # +1 por causa do bias
        populacao = criar_cromossomos(self.qtd_cromossomos, qtd_genes, self.minimo, self.maximo)

        self.fitness = -np.inf
        self.acuracia = 0.0
        self.historico = {"melhor": [], "medio": [], "acuracia": []}

        for geracao in range(self.geracoes):
            fitnesses = self._calcular_fitness(populacao, X, y)
            percentuais = fitness_percentual(fitnesses)

            idx_melhor = int(np.argmax(fitnesses))
            if fitnesses[idx_melhor] > self.fitness:
                self.fitness = float(fitnesses[idx_melhor])
                self.cromossomo = populacao[idx_melhor].copy()
                # como o cromossomo já está no self, dá para usar o próprio prever()
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

            # seleção -> cruzamento -> mutação -> avaliação -> substituição
            pai, mae = selecionar_pais_roleta(populacao, percentuais)
            filhos = cruzar_um_ponto(pai, mae, self.qtd_filhos)
            filhos = mutar(filhos, self.qtd_genes_mutados)
            fitnesses_filhos = self._calcular_fitness(filhos, X, y)
            populacao = atualizar_populacao(populacao, fitnesses, filhos, fitnesses_filhos, self.qtd_substituicoes)

        return self

    def prever(self, X: np.ndarray) -> np.ndarray:
        """
        Classifica dados novos com o melhor cromossomo encontrado.

        Parâmetros:
            X -> array (features,) com um registro ou (amostras, features) com vários

        Retorna:
            array de inteiros com a classe prevista de cada registro
        """

        # no lugar de conferir as chaves do dict, só confere se já treinou
        if self.cromossomo is None:
            raise ValueError("o modelo ainda não foi treinado, chame treinar() antes")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        bias = self.cromossomo[0]
        genes = self.cromossomo[1:]
        q = X @ genes + bias

        if self.thresholds is None:
            return np.where(q >= 0, 1, 0)
        return np.digitize(q, self.thresholds)

    def __repr__(self) -> str:
        """O que aparece quando você dá print(modelo)."""
        status = "não treinado" if self.cromossomo is None else f"fitness={self.fitness:.4f}"
        return f"ClassificadorGenetico(qtd_cromossomos={self.qtd_cromossomos}, geracoes={self.geracoes}, {status})"


# as funções antigas continuariam existindo, usando a classe por dentro,
# para não quebrar quem já usa a versão 1.0.0
def rodar(X: np.ndarray, y: np.ndarray, **parametros) -> dict:
    modelo = ClassificadorGenetico(**parametros).treinar(X, y)
    return {
        "cromossomo": modelo.cromossomo,
        "fitness": modelo.fitness,
        "acuracia": modelo.acuracia,
        "geracao": modelo.geracao,
        "thresholds": modelo.thresholds,
        "historico": modelo.historico,
    }


if __name__ == "__main__":
    from algoritmo_genetico import dividir_treino_teste, ler_excel

    X, y = ler_excel("exemplos/exemplo_1_binario/dados_binario_1.xlsx")
    X_treino, X_teste, y_treino, y_teste = dividir_treino_teste(X, y)

    modelo = ClassificadorGenetico(qtd_cromossomos=10, geracoes=50, seed=42, verbose=False)
    print(modelo)

    modelo.treinar(X_treino, y_treino)
    print(modelo)

    previsoes = modelo.prever(X_teste)
    print("acurácia no teste:", np.mean(previsoes == y_teste))
