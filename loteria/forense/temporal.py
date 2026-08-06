"""Testes temporais sobre as séries indicadoras 0/1 de cada dezena:
Ljung-Box, runs de Wald-Wolfowitz, espectro (teste g de Fisher) e gaps.

Sob H0 a série indicadora de uma dezena é Bernoulli(k/N) iid ao longo dos
concursos, então cada dezena gera um teste e a família absorve a
multiplicidade via BH.
"""

import math

import numpy as np
from scipy import stats

from .resultado import ResultadoTeste


def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
    x = x - x.mean()
    var = float((x * x).sum())
    if var == 0:
        return np.zeros(nlags)
    return np.array([(x[:-l] * x[l:]).sum() / var for l in range(1, nlags + 1)])


def ljung_box(series: dict[str, np.ndarray], nlags: int = 20) -> list[ResultadoTeste]:
    resultados = []
    for nome, x in series.items():
        n = len(x)
        L = min(nlags, n // 5)
        r = _acf(x.astype(float), L)
        lags = np.arange(1, L + 1)
        q = float(n * (n + 2) * ((r ** 2) / (n - lags)).sum())
        p = float(stats.chi2.sf(q, df=L))
        resultados.append(ResultadoTeste(
            familia="ljung_box", teste=nome,
            estatistica=q, p_valor=p, efeito=float(np.abs(r).max()),
            detalhe={"lags": L},
        ))
    return resultados


def runs_wald_wolfowitz(series: dict[str, np.ndarray]) -> list[ResultadoTeste]:
    resultados = []
    for nome, x in series.items():
        n1 = int(x.sum())
        n0 = len(x) - n1
        if n1 < 10 or n0 < 10:
            continue  # aproximação normal não confiável
        runs = 1 + int((x[1:] != x[:-1]).sum())
        n = n0 + n1
        mu = 2 * n1 * n0 / n + 1
        var = 2 * n1 * n0 * (2 * n1 * n0 - n) / (n ** 2 * (n - 1))
        z = (runs - mu) / math.sqrt(var)
        p = float(2 * stats.norm.sf(abs(z)))
        resultados.append(ResultadoTeste(
            familia="runs", teste=nome,
            estatistica=float(z), p_valor=p, efeito=float(abs(z)),
            detalhe={"runs": runs, "esperado": mu},
        ))
    return resultados


def _fisher_g_p(g: float, q: int) -> float:
    """P(G > g) exato do teste g de Fisher para o maior pico do periodograma."""
    if g <= 0:
        return 1.0
    jmax = min(q, int(1.0 / g))
    total = 0.0
    for j in range(1, jmax + 1):
        termo = (math.lgamma(q + 1) - math.lgamma(j + 1) - math.lgamma(q - j + 1)
                 + (q - 1) * math.log1p(-j * g))
        total += (-1) ** (j - 1) * math.exp(termo)
    return float(min(max(total, 0.0), 1.0))


def espectral_fisher(series: dict[str, np.ndarray]) -> list[ResultadoTeste]:
    """FFT da série indicadora; teste g de Fisher detecta pico periódico
    dominante acima do ruído branco."""
    resultados = []
    for nome, x in series.items():
        xc = x.astype(float) - x.mean()
        espectro = np.abs(np.fft.rfft(xc)) ** 2
        # exclui frequência zero e, se n par, a de Nyquist (distribuição difere)
        fim = len(espectro) - 1 if len(x) % 2 == 0 else len(espectro)
        pico = espectro[1:fim]
        if len(pico) < 8:
            continue
        soma = float(pico.sum())
        if soma == 0:
            continue
        g = float(pico.max() / soma)
        q = len(pico)
        p = _fisher_g_p(g, q)
        freq_pico = int(np.argmax(pico)) + 1
        resultados.append(ResultadoTeste(
            familia="espectral", teste=nome,
            estatistica=g, p_valor=p, efeito=g,
            detalhe={"periodo_concursos": round(len(x) / freq_pico, 1)},
        ))
    return resultados


def gaps_geometrico(series: dict[str, np.ndarray], p_sucesso: float,
                    minimo_esperado: float = 5.0) -> list[ResultadoTeste]:
    """Gaps entre aparições de cada dezena vs. Geométrica(k/N): qui-quadrado
    de aderência com caudas agrupadas, agregado por dezena e agrupado
    ('pooled') no teste principal."""
    resultados = []
    todos_gaps: list[np.ndarray] = []
    for nome, x in series.items():
        idx = np.flatnonzero(x)
        if len(idx) >= 2:
            todos_gaps.append(np.diff(idx))  # gap 1 = saiu no concurso seguinte

    if not todos_gaps:
        return resultados
    gaps = np.concatenate(todos_gaps)
    resultados.append(_gof_geometrico(gaps, p_sucesso, "agrupado (todas as dezenas)",
                                      minimo_esperado))

    # por dezena: z-teste do gap médio (média geométrica = 1/p)
    for nome, x in series.items():
        idx = np.flatnonzero(x)
        if len(idx) < 30:
            continue
        g = np.diff(idx)
        mu = 1.0 / p_sucesso
        sigma = math.sqrt((1 - p_sucesso) / p_sucesso ** 2)
        z = (g.mean() - mu) / (sigma / math.sqrt(len(g)))
        resultados.append(ResultadoTeste(
            familia="gaps", teste=f"gap médio {nome}",
            estatistica=float(z), p_valor=float(2 * stats.norm.sf(abs(z))),
            efeito=float(g.mean() / mu - 1),
            detalhe={"n_gaps": int(len(g)), "gap_medio": float(g.mean())},
        ))
    return resultados


def _gof_geometrico(gaps: np.ndarray, p: float, nome: str,
                    minimo_esperado: float) -> ResultadoTeste:
    n = len(gaps)
    # bins 1,2,...,t*, cauda >= t* com esperado >= minimo_esperado
    t = 1
    esperados, limites = [], []
    while True:
        prob = (1 - p) ** (t - 1) * p
        cauda = (1 - p) ** t
        if n * cauda < minimo_esperado or t > 500:
            esperados.append(n * ((1 - p) ** (t - 1)))  # cauda >= t
            limites.append(t)
            break
        esperados.append(n * prob)
        limites.append(t)
        t += 1
    observados = np.zeros(len(limites))
    for i, lim in enumerate(limites[:-1]):
        observados[i] = int((gaps == lim).sum())
    observados[-1] = int((gaps >= limites[-1]).sum())
    esperados_arr = np.array(esperados)
    x2 = float(((observados - esperados_arr) ** 2 / esperados_arr).sum())
    df = len(limites) - 1
    return ResultadoTeste(
        familia="gaps", teste=nome,
        estatistica=x2, p_valor=float(stats.chi2.sf(x2, df=df)),
        efeito=float(np.sqrt(x2 / n)),
        detalhe={"df": df, "n_gaps": int(n), "gap_maximo": int(gaps.max())},
    )
