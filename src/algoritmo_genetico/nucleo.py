"""Loop principal do algoritmo genético: treino (treinar) e previsão (prever)."""

import warnings

import numpy as np

from .atualizacao import atualizar_populacao
from .criacao import criar_cromossomos
from .cruzamento import cruzar_um_ponto
from .fitness import fitness_binario, fitness_ordinal, fitness_percentual
from .mutacao import mutar
from .selecao import selecionar_pais_roleta


def treinar(
    X: np.ndarray,
    y: np.ndarray,
    qtd_cromossomos: int = 30,
    qtd_genes_mutados: int = 2,
    qtd_filhos: int = 16,
    qtd_substituicoes: int | None = None,
    geracoes: int = 1000,
    fitness_alvo: float = 1.0,
    thresholds: np.ndarray | None = None,
    minimo: float = -1.0,
    maximo: float = 1.0,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    
    """
    Treina o modelo com o algoritmo genético completo: cria a população e repete
    fitness -> seleção -> cruzamento -> mutação -> atualização até atingir
    o fitness alvo ou acabar as gerações.

    Parâmetros:
        X -> array com os dados
        y -> vetor com a classe certa de cada amostra
        qtd_cromossomos -> tamanho da população
        qtd_genes_mutados -> quantos genes mudar em cada filho na mutação
        qtd_filhos -> quantos filhos gerar por geração
        qtd_substituicoes -> quantos piores pais serão trocados pelos melhores filhos
                             (None = metade de qtd_filhos, limitado a qtd_cromossomos)
        geracoes -> número máximo de gerações
        fitness_alvo -> para quando o melhor fitness chegar nesse valor
                        (1.0 = só para se acertar tudo, ou seja, roda todas as gerações)
        thresholds -> None para 2 classes (binário), ou array com K-1 pontos de corte
                      em ordem crescente para K classes (ordinal)
        minimo -> valor mínimo dos genes na criação da população
        maximo -> valor máximo dos genes na criação da população
        seed -> semente para reprodutibilidade (None = aleatório a cada rodada)
        verbose -> se True, mostra o melhor fitness e a acurácia de cada geração

    Retorna:
        dict com:
            "cromossomo" -> melhor cromossomo encontrado
            "fitness" -> fitness do melhor cromossomo
            "acuracia" -> fração das amostras de X que o melhor cromossomo acerta
            "geracao" -> geração em que parou
            "thresholds" -> thresholds usados (None se binário)
            "historico" -> dict com evolução do melhor fitness, da média e da
                           acurácia do melhor por geração
    """

    if not isinstance(X, np.ndarray):
        raise TypeError("O parâmetro X deve ser do tipo ndarray")

    if not isinstance(y, np.ndarray):
        raise TypeError("O parâmetro y deve ser do tipo ndarray")

    if X.ndim != 2:
        raise ValueError(f"X deve ser matriz 2D (amostras, features), recebeu {X.ndim} dimensões")

    if y.ndim != 1:
        raise ValueError(f"y deve ser vetor 1D, recebeu {y.ndim} dimensões")

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X tem {X.shape[0]} amostras, mas y tem {y.shape[0]} valores")

    if not isinstance(qtd_cromossomos, int):
        raise TypeError("qtd_cromossomos deve ser um número inteiro")

    if qtd_cromossomos <= 0:
        raise ValueError(f"qtd_cromossomos deve ser >= 1, recebeu {qtd_cromossomos}")

    if not isinstance(qtd_genes_mutados, int):
        raise TypeError("qtd_genes_mutados deve ser um número inteiro")

    if qtd_genes_mutados < 0:
        raise ValueError(f"qtd_genes_mutados deve ser >= 0, recebeu {qtd_genes_mutados}")

    if not isinstance(qtd_filhos, int):
        raise TypeError("qtd_filhos deve ser um número inteiro")

    if qtd_filhos <= 0:
        raise ValueError(f"qtd_filhos deve ser >= 1, recebeu {qtd_filhos}")

    if qtd_substituicoes is None:
        qtd_substituicoes = min(max(qtd_filhos // 2, 1), qtd_cromossomos)

    if not isinstance(qtd_substituicoes, int):
        raise TypeError("qtd_substituicoes deve ser um número inteiro ou None")

    if qtd_substituicoes < 0 or qtd_substituicoes > min(qtd_cromossomos, qtd_filhos):
        raise ValueError(f"qtd_substituicoes deve estar entre 0 e {min(qtd_cromossomos, qtd_filhos)} (menor valor entre qtd_cromossomos e qtd_filhos), recebeu {qtd_substituicoes}")

    if not isinstance(geracoes, int):
        raise TypeError("geracoes deve ser um número inteiro")

    if geracoes <= 0:
        raise ValueError(f"geracoes deve ser >= 1, recebeu {geracoes}")

    if not isinstance(fitness_alvo, (int, float)):
        raise TypeError("fitness_alvo deve ser um número")

    if fitness_alvo < 0 or fitness_alvo > 1:
        raise ValueError(f"fitness_alvo deve estar entre 0 e 1, recebeu {fitness_alvo}")

    if not isinstance(minimo, (int, float)):
        raise TypeError("minimo deve ser um número")

    if not isinstance(maximo, (int, float)):
        raise TypeError("maximo deve ser um número")

    if minimo >= maximo:
        raise ValueError(f"minimo ({minimo}) deve ser menor que maximo ({maximo})")

    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise TypeError("seed deve ser um número inteiro ou None")

    if not isinstance(verbose, bool):
        raise TypeError("verbose deve ser True ou False")

    if thresholds is not None and not isinstance(thresholds, np.ndarray):
        raise TypeError("thresholds deve ser do tipo ndarray ou None")

    if seed is not None:
        np.random.seed(seed)

    # converte uma vez só: X @ pesos com X inteiro (ex: saída do ler_csv) é ~6x mais lento que com float
    X = X.astype(float)

    qtd_genes = X.shape[1] + 1 #adicionei o +1 por causa do bias


    if thresholds is None:
        def calcular(crom):
            return fitness_binario(crom, X, y)
    else:
        thresholds = np.asarray(thresholds)

        def calcular(crom):
            return fitness_ordinal(crom, X, y, thresholds)

    populacao = criar_cromossomos(qtd_cromossomos, qtd_genes, minimo, maximo)

    melhor_cromossomo = None
    melhor_fitness = -np.inf
    melhor_acuracia = 0.0
    geracao_final = 0
    historico = {"melhor": [], "medio": [], "acuracia": []}

    for geracao in range(geracoes):
        fitnesses = calcular(populacao)
        percentuais = fitness_percentual(fitnesses)

        # pega o melhor da geração atual
        idx_melhor = int(np.argmax(fitnesses))
        fitness_atual = float(fitnesses[idx_melhor])

        # atualiza o melhor global, se melhorou
        if fitness_atual > melhor_fitness:
            melhor_fitness = fitness_atual
            melhor_cromossomo = populacao[idx_melhor].copy()

            # acurácia = acertos / total, só recalcula quando o melhor muda
            y_pred = prever({"cromossomo": melhor_cromossomo, "thresholds": thresholds}, X)
            melhor_acuracia = float(np.mean(y_pred == y))

        geracao_final = geracao + 1
        historico["melhor"].append(melhor_fitness)
        historico["medio"].append(float(fitnesses.mean()))
        historico["acuracia"].append(melhor_acuracia)

        if verbose:
            print(f"Geração {geracao_final:>4} | melhor fitness: {melhor_fitness:.4f} | acurácia: {melhor_acuracia:.2%}")

        # critério de parada por fitness alvo
        if melhor_fitness >= fitness_alvo:
            if verbose:
                print(f"\nFitness alvo {fitness_alvo} atingido na geração {geracao_final}.")
            break

        # seleção -> cruzamento -> mutação -> avaliação -> substituição
        pai, mae = selecionar_pais_roleta(populacao, percentuais)
        filhos = cruzar_um_ponto(pai, mae, qtd_filhos)
        filhos = mutar(filhos, qtd_genes_mutados)
        fitnesses_filhos = calcular(filhos)
        populacao = atualizar_populacao(
            populacao, fitnesses,
            filhos, fitnesses_filhos,
            qtd_substituicoes,
        )

    return {
        "cromossomo": melhor_cromossomo,
        "fitness": melhor_fitness,
        "acuracia": melhor_acuracia,
        "geracao": geracao_final,
        "thresholds": thresholds,
        "historico": historico,
    }


def rodar(*args, **kwargs) -> dict:
    """
    Nome antigo de treinar(), mantido para não quebrar código existente.
    Será removido numa versão futura: use treinar().
    """
    warnings.warn(
        "rodar() foi renomeada para treinar() e será removida numa versão futura",
        DeprecationWarning,
        stacklevel=2,
    )
    return treinar(*args, **kwargs)


def prever(modelo: dict, X: np.ndarray) -> np.ndarray:
    """
    Usa o melhor cromossomo encontrado por treinar() para classificar dados novos.

    Parâmetros:
        modelo -> dict retornado por treinar()
        X -> array (features,) com um registro ou (amostras, features) com vários

    Retorna:
        array de inteiros com a classe prevista de cada registro
    """

    if not isinstance(modelo, dict):
        raise TypeError("O parâmetro modelo deve ser um dict (retornado por treinar())")

    if "cromossomo" not in modelo:
        raise ValueError("modelo inválido: falta a chave 'cromossomo'")

    if "thresholds" not in modelo:
        raise ValueError("modelo inválido: falta a chave 'thresholds'")

    if modelo["cromossomo"] is None:
        raise ValueError("modelo não foi treinado (cromossomo é None)")

    if not isinstance(modelo["cromossomo"], np.ndarray):
        raise TypeError("modelo['cromossomo'] deve ser do tipo ndarray")

    if modelo["thresholds"] is not None and not isinstance(modelo["thresholds"], np.ndarray):
        raise TypeError("modelo['thresholds'] deve ser do tipo ndarray ou None")

    if not isinstance(X, np.ndarray):
        raise TypeError("O parâmetro X deve ser do tipo ndarray")

    if X.ndim not in (1, 2):
        raise ValueError(f"X deve ser 1D (um registro) ou 2D (vários registros), recebeu {X.ndim}D")

    cromossomo = modelo["cromossomo"]
    thresholds = modelo["thresholds"]

    bias = cromossomo[0]
    genes = cromossomo[1:]

    # se for um único registro (1D), transforma em matriz (1, features) para operar igual
    if X.ndim == 1:
        X = X.reshape(1, -1)

    # confere que o número de features bate com o cromossomo
    if X.shape[1] != len(genes):
        raise ValueError(
            f"X tem {X.shape[1]} features, mas o cromossomo foi treinado com {len(genes)} features"
        )

    q = X @ genes + bias

    if thresholds is None:
        return np.where(q >= 0, 1, 0)
    return np.digitize(q, thresholds)