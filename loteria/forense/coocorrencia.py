"""Coocorrência de pares e trincas vs. o esperado hipergeométrico.

Sob H0, a probabilidade de um par {i,j} sair junto num concurso é
k(k-1)/(N(N-1)) e de uma trinca é k(k-1)(k-2)/(N(N-1)(N-2)); a contagem em m
concursos é Binomial(m, p). O p-valor bicaudal exato é vetorizado via cdf/sf.

A correção BH é calculada AQUI, sobre o vetor completo de p-valores da
família (todos os C(N,2) pares / C(N,3) trincas, incluindo os de contagem
zero) — só as linhas de menor q viram registro individual no relatório, mas o
denominador da correção nunca é truncado.

Para o Super Sete a noção análoga é independência entre colunas: qui-quadrado
de contingência 10×10 para cada par de colunas.
"""

from itertools import combinations

import numpy as np
from scipy import stats

from .resultado import ResultadoTeste, bh_qvalores


def _p_binomial_bicaudal(contagens: np.ndarray, m: int, p: float) -> np.ndarray:
    baixo = stats.binom.cdf(contagens, m, p)
    alto = stats.binom.sf(contagens - 1, m, p)
    return np.minimum(1.0, 2.0 * np.minimum(baixo, alto))


def pares(indicadores: np.ndarray, k: int, top_n: int = 30) -> list[ResultadoTeste]:
    m, n_univ = indicadores.shape
    x = indicadores.astype(np.int32)
    co = x.T @ x
    iu, ju = np.triu_indices(n_univ, k=1)
    contagens = co[iu, ju]
    p_h0 = k * (k - 1) / (n_univ * (n_univ - 1))
    ps = _p_binomial_bicaudal(contagens, m, p_h0)
    qs = bh_qvalores(ps)
    esperado = m * p_h0

    nomes = [f"par {iu[t] + 1:02d}-{ju[t] + 1:02d}" for t in range(len(ps))]
    return _selecionar("coocorrencia_pares", nomes, contagens, ps, qs,
                       esperado, top_n)


def trincas(sorteios_idx: list[np.ndarray], n_univ: int, m: int, k: int,
            top_n: int = 30) -> list[ResultadoTeste]:
    """sorteios_idx: índices 0-based das dezenas de cada concurso."""
    contagem: dict[tuple, int] = {}
    for dezenas in sorteios_idx:
        for t in combinations(sorted(dezenas.tolist()), 3):
            contagem[t] = contagem.get(t, 0) + 1
    p_h0 = (k * (k - 1) * (k - 2)) / (n_univ * (n_univ - 1) * (n_univ - 2))
    esperado = m * p_h0

    total_trincas = n_univ * (n_univ - 1) * (n_univ - 2) // 6
    chaves = list(contagem.keys())
    vistos = np.array([contagem[c] for c in chaves], dtype=np.int64)
    n_zero = total_trincas - len(chaves)
    # trincas nunca sorteadas também são evidência e entram na correção
    contagens = np.concatenate([vistos, np.zeros(n_zero, dtype=np.int64)])
    ps = _p_binomial_bicaudal(contagens, m, p_h0)
    qs = bh_qvalores(ps)

    nomes = [f"trinca {i + 1:02d}-{j + 1:02d}-{l + 1:02d}" for i, j, l in chaves]
    nomes += ["trinca (nunca sorteada)"] * n_zero
    return _selecionar("coocorrencia_trincas", nomes, contagens, ps, qs,
                       esperado, top_n)


def _selecionar(familia, nomes, contagens, ps, qs, esperado, top_n):
    """Reporta as top_n linhas de menor q; o resto vira uma linha-síntese.
    Os q_valores já vêm corrigidos sobre a família completa."""
    ordem = np.argsort(qs, kind="stable")
    resultados = []
    for pos in ordem[:top_n]:
        resultados.append(ResultadoTeste(
            familia=familia, teste=nomes[pos],
            estatistica=float(contagens[pos]),
            p_valor=float(ps[pos]), q_valor=float(qs[pos]),
            efeito=float(contagens[pos] / esperado - 1),
            detalhe={"esperado": round(esperado, 2)},
        ))
    resto = ordem[top_n:]
    if len(resto):
        resultados.append(ResultadoTeste(
            familia=familia, teste=f"demais {len(resto)} combinações (síntese)",
            estatistica=float("nan"),
            p_valor=float(ps[resto].min()), q_valor=float(qs[resto].min()),
            efeito=None,
            detalhe={"n_agregados": int(len(resto)),
                     "q_minimo_do_resto": float(qs[resto].min())},
        ))
    return resultados


def independencia_colunas(digitos: np.ndarray) -> list[ResultadoTeste]:
    """Super Sete: qui-quadrado de contingência 10×10 por par de colunas."""
    m, n_col = digitos.shape
    resultados = []
    for a, b in combinations(range(n_col), 2):
        tabela = np.zeros((10, 10))
        np.add.at(tabela, (digitos[:, a], digitos[:, b]), 1)
        x2, p, df, _ = stats.chi2_contingency(tabela)
        v = float(np.sqrt(x2 / (m * 9)))  # V de Cramér
        resultados.append(ResultadoTeste(
            familia="coocorrencia_pares", teste=f"colunas {a + 1}×{b + 1}",
            estatistica=float(x2), p_valor=float(p), efeito=v,
            detalhe={"df": int(df)},
        ))
    return resultados
