# Algoritmo genético para classificação

**Fellipe Almassy** · Novembro de 2026

Este documento explica como um algoritmo genético pode treinar um classificador linear. Em vez de usar gradiente, o método evolui uma população de soluções candidatas: as melhores têm mais chance de se reproduzir, seus descendentes sofrem mutações e substituem as piores soluções. O texto descreve cada etapa do algoritmo, com a notação matemática e um exemplo numérico, e indica a função correspondente da biblioteca `algoritmo-genetico`.

## Objetivo

Partimos de uma base de dados rotulada, formada por lições (amostras). Cada lição tem um vetor de *features* e uma classe conhecida. O objetivo do algoritmo genético é encontrar os coeficientes de uma função linear que, a partir das features, indique a classe de cada lição. Depois do treinamento, essa função pode classificar dados novos.

## Notação

| Símbolo | Significado |
|---|---|
| $N$ | número de lições da base |
| $n$ | número de features |
| $K$ | número de classes, numeradas de $0$ a $K-1$ |
| $\mathbf{x} = (x_1, \dots, x_n)$ | features de uma lição |
| $y$ | classe real de uma lição |
| $m$ | número de cromossomos da população |
| $L = n + 1$ | número de genes de cada cromossomo |
| $h$ | número de filhos gerados por geração |
| $r$ | número de genes alterados em cada filho |
| $s$ | número de cromossomos substituídos por geração |

## Representação da solução

Cada **cromossomo** é um vetor de números reais que representa uma solução candidata. Ele é formado por um **bias** $b$ e por um **gene** $g_j$ para cada feature:

$$
\mathbf{c} = (b, g_1, g_2, \dots, g_n)
$$

Com esses valores, o cromossomo calcula uma pontuação $q$ para qualquer lição:

$$
q(\mathbf{x}) = b + g_1 x_1 + g_2 x_2 + \dots + g_n x_n = b + \sum_{j=1}^{n} g_j x_j
$$

Cada gene indica o peso de uma feature na decisão. O bias desloca a pontuação de todas as lições pelo mesmo valor. Sem ele, uma lição com todas as features iguais a zero teria sempre $q = 0$, e a fronteira de decisão seria obrigada a passar pela origem.

Na classificação binária, a pontuação é convertida em classe pelo sinal:

$$
\hat{y} =
\begin{cases}
1, & \text{se } q(\mathbf{x}) \geq 0 \\
0, & \text{se } q(\mathbf{x}) < 0
\end{cases}
$$

Na classificação ordinal, as $K$ classes têm uma ordem (por exemplo, ruim < médio < bom). Definimos $K-1$ pontos de corte fixos $t_1 < t_2 < \dots < t_{K-1}$, escolhidos pelo usuário e que não evoluem durante o treino. Com as convenções $t_0 = -\infty$ e $t_K = +\infty$, a classe prevista é:

$$
\hat{y} = k \quad \text{tal que} \quad t_k \leq q(\mathbf{x}) < t_{k+1}
$$

## O ciclo do algoritmo

1. Criar a população inicial.
2. Calcular o fitness de cada cromossomo.
3. Selecionar um pai e uma mãe por roleta.
4. Gerar $h$ filhos por cruzamento.
5. Aplicar mutação aos filhos.
6. Substituir os $s$ piores cromossomos pelos $s$ melhores filhos e voltar à etapa 2.

O ciclo termina quando o melhor fitness encontrado atinge o valor alvo ou quando acaba o número máximo de gerações. O resultado é o melhor cromossomo visto em todo o treino, guardado à parte, e não necessariamente um cromossomo da última população.

### Etapa 1: criação da população

A população inicial é uma matriz $P$ com $m$ linhas e $L$ colunas. Cada linha é um cromossomo e cada coluna é um gene, sendo a primeira coluna a do bias:

$$
P =
\begin{pmatrix}
b_1 & g_{1,1} & g_{1,2} & \cdots & g_{1,n} \\
b_2 & g_{2,1} & g_{2,2} & \cdots & g_{2,n} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
b_m & g_{m,1} & g_{m,2} & \cdots & g_{m,n}
\end{pmatrix}
$$

Todos os valores são sorteados de maneira uniforme em um intervalo, por padrão $[-1, 1]$. Os limites são parâmetros da função e podem ser alterados.

### Etapa 2: cálculo do fitness

O fitness é a nota de um cromossomo: quanto melhor ele classifica as lições da base, maior a nota. A medida mais direta seria a acurácia, isto é, a fração de acertos. Ela, porém, é enganosa quando as classes são desbalanceadas. Em uma base com 95 lições da classe 1 e 5 da classe 0, um cromossomo que responde sempre 1 tem 95% de acurácia sem ter aprendido nada.

Por isso, medimos o recall de cada classe, ou seja, a fração de lições dessa classe que o cromossomo acertou:

$$
R_k = \frac{\text{acertos na classe } k}{\text{total de lições da classe } k}
$$

O fitness é o produto dos recalls:

$$
F = \prod_{k=0}^{K-1} R_k
$$

Assim, $0 \leq F \leq 1$, e $F = 0$ sempre que o cromossomo erra todas as lições de alguma classe. No caso acima, o cromossomo que responde sempre 1 tem $R_1 = 1$ e $R_0 = 0$, logo $F = 0$. O resultado depende apenas da fração de acertos dentro de cada classe, e não de quantas lições cada uma tem.

**Exemplo.** Considere a base abaixo e o cromossomo $\mathbf{c} = (2, 3, -4, 1, -1)$. Os valores são inteiros e estão fora do intervalo padrão apenas para facilitar as contas. A pontuação fica:

$$
q(\mathbf{x}) = 2 + 3x_1 - 4x_2 + 1x_3 - 1x_4
$$

| | $x_1$ | $x_2$ | $x_3$ | $x_4$ | Classe |
|---|---|---|---|---|---|
| Lição 1 | 1 | 0 | 1 | 1 | 1 |
| Lição 2 | 1 | 1 | 0 | 0 | 1 |
| Lição 3 | 1 | 0 | 0 | 1 | 0 |
| Lição 4 | 0 | 1 | 1 | 1 | 0 |

Aplicando a pontuação a cada lição e convertendo o resultado em classe pela regra binária:

| | Cálculo | $q$ | Prevista | Real | Resultado |
|---|---|---|---|---|---|
| Lição 1 | $2 + 3 \cdot 1 - 4 \cdot 0 + 1 \cdot 1 - 1 \cdot 1$ | $5$ | 1 | 1 | acerto |
| Lição 2 | $2 + 3 \cdot 1 - 4 \cdot 1 + 1 \cdot 0 - 1 \cdot 0$ | $1$ | 1 | 1 | acerto |
| Lição 3 | $2 + 3 \cdot 1 - 4 \cdot 0 + 1 \cdot 0 - 1 \cdot 1$ | $4$ | 1 | 0 | erro |
| Lição 4 | $2 + 3 \cdot 0 - 4 \cdot 1 + 1 \cdot 1 - 1 \cdot 1$ | $-2$ | 0 | 0 | acerto |

O cromossomo acertou 3 das 4 lições, uma acurácia de 75%. Por classe, temos:

$$
R_1 = \frac{2}{2} = 1, \qquad R_0 = \frac{1}{2} = 0{,}5, \qquad F = R_1 \cdot R_0 = 0{,}5
$$

### Etapa 3: seleção por roleta

Os fitnesses são normalizados em probabilidades que somam 1:

$$
p_i = \frac{F_i}{\sum_{l=1}^{m} F_l}
$$

Se todos os fitnesses forem zero, todos os cromossomos recebem a mesma chance, $p_i = 1/m$.

Para girar a roleta, sorteamos $u$ uniformemente em $[0, 1)$ e escolhemos o primeiro cromossomo cuja probabilidade acumulada ultrapassa $u$. O sorteio é feito duas vezes, uma para o pai e outra para a mãe, de modo que os dois podem ser o mesmo cromossomo. A cada geração é escolhido um único casal.

**Exemplo.** Com fitnesses 0,5, 0,25 e 0,25, as probabilidades são as mesmas, e as acumuladas são 0,5, 0,75 e 1. Se $u = 0{,}6$, o sorteado é o segundo cromossomo.

### Etapa 4: cruzamento de um ponto

Para cada um dos $h$ filhos, sorteamos um ponto de corte $c \in \{1, \dots, L-1\}$. O filho recebe os $c$ primeiros genes do pai e os demais genes da mãe:

$$
\mathbf{c}^{\text{filho}} = (\mathbf{c}^{\text{pai}}_{1}, \dots, \mathbf{c}^{\text{pai}}_{c}, \mathbf{c}^{\text{mãe}}_{c+1}, \dots, \mathbf{c}^{\text{mãe}}_{L})
$$

Como o corte nunca fica nas pontas, o filho sempre herda ao menos um gene de cada um.

**Exemplo.** Com pai $(2, 3, -4, 1, -1)$, mãe $(-1, 0{,}5, 2, -3, 1)$ e $c = 2$, o filho é $(2, 3, 2, -3, 1)$.

### Etapa 5: mutação

Em cada filho, sorteamos $r$ posições diferentes e trocamos o gene de cada uma por um novo valor uniforme em $[-1, 1]$. A mutação traz para a população valores que não existiam nos pais e, por isso, ajuda a manter a diversidade. Na implementação atual, esse intervalo é fixo e não acompanha os limites escolhidos na etapa 1.

**Exemplo.** Se a posição sorteada for a do quarto gene e o novo valor for 0,4, o filho $(2, 3, 2, -3, 1)$ passa a ser $(2, 3, 2, 0{,}4, 1)$.

### Etapa 6: atualização da população

Calculamos o fitness dos filhos mutados. Os $s$ cromossomos de menor fitness da população saem e os $s$ filhos de maior fitness entram em seu lugar, com $0 \leq s \leq \min(m, h)$. O restante da população passa intacto para a próxima geração, e os demais filhos são descartados.

A troca não compara filhos e pais: um filho pode entrar mesmo com fitness menor que o do cromossomo que substituiu. Quando $s < m$, os melhores cromossomos da população são preservados, já que apenas os piores saem.

## Parâmetros

Parâmetros da função `treinar` e seus valores padrão:

| Símbolo | Parâmetro | Padrão |
|---|---|---|
| $m$ | `qtd_cromossomos` | 30 |
| $r$ | `qtd_genes_mutados` | 2 |
| $h$ | `qtd_filhos` | 16 |
| $s$ | `qtd_substituicoes` | metade de $h$ (8) |
| — | `geracoes` | 1000 |
| — | `fitness_alvo` | 1,0 |
| $t_1, \dots, t_{K-1}$ | `thresholds` | nenhum (binário) |
| — | `minimo`, `maximo` | $-1$ e $1$ |

## Correspondência com o código

Cada etapa está implementada em `src/algoritmo_genetico/`:

| Etapa | Arquivo | Função |
|---|---|---|
| 1. Criação | `criacao.py` | `criar_cromossomos` |
| 2. Fitness | `fitness.py` | `fitness_binario`, `fitness_ordinal` |
| 3. Seleção | `fitness.py`, `selecao.py` | `fitness_percentual`, `selecionar_pais_roleta` |
| 4. Cruzamento | `cruzamento.py` | `cruzar_um_ponto` |
| 5. Mutação | `mutacao.py` | `mutar` |
| 6. Atualização | `atualizacao.py` | `atualizar_populacao` |
| Ciclo completo | `nucleo.py` | `treinar` |
| Classificação | `nucleo.py` | `prever` |
