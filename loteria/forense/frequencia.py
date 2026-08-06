"""Testes de frequência: qui-quadrado (global e janelas), KS, entropia/G.

Nos jogos sem reposição (k dezenas de N por concurso) as contagens por dezena
têm correlação negativa dentro do concurso; o qui-quadrado clássico fica
descalibrado. Usamos o ajuste de Joe (1993, "Tests of uniformity for sets of
lotto numbers"): X²_aj = X² · (N-1)/(N-k) ~ χ²(N-1). Para o Super Sete
(1 dígito de 10 por coluna, com reposição entre concursos) o fator é 1.
A calibração é verificada empiricamente pelo controle negativo da Fase 3.
"""

import numpy as np
from scipy import stats

from .resultado import ResultadoTeste


def _chi2_ajustado(contagens: np.ndarray, esperado: float, n_universo: int,
                   k: int) -> tuple[float, float, float]:
    """Retorna (estatística ajustada, p, efeito w de Cohen)."""
    x2 = float(((contagens - esperado) ** 2 / esperado).sum())
    ajuste = (n_universo - 1) / (n_universo - k)
    x2_aj = x2 * ajuste
    p = float(stats.chi2.sf(x2_aj, df=n_universo - 1))
    w = float(np.sqrt(x2 / contagens.sum()))
    return x2_aj, p, w


def chi2_global(indicadores: np.ndarray, k: int) -> ResultadoTeste:
    """indicadores: matriz (concursos × universo) de 0/1."""
    m, n_univ = indicadores.shape
    contagens = indicadores.sum(axis=0)
    x2, p, w = _chi2_ajustado(contagens, m * k / n_univ, n_univ, k)
    return ResultadoTeste(
        familia="chi2_frequencia", teste="global",
        estatistica=x2, p_valor=p, efeito=w,
        detalhe={"df": n_univ - 1, "concursos": m,
                 "min_contagem": int(contagens.min()),
                 "max_contagem": int(contagens.max())},
    )


def chi2_janelas(indicadores: np.ndarray, k: int, largura: int = 250,
                 passo: int = 125) -> list[ResultadoTeste]:
    """Janelas móveis para captar mudanças de regime (troca de bolas/globo).

    Cada janela vira um teste da família 'chi2_janelas'; a correção BH da
    família absorve a multiplicidade de janelas.
    """
    m, n_univ = indicadores.shape
    if m < largura:
        return []
    resultados = []
    for inicio in range(0, m - largura + 1, passo):
        bloco = indicadores[inicio:inicio + largura]
        contagens = bloco.sum(axis=0)
        x2, p, w = _chi2_ajustado(contagens, largura * k / n_univ, n_univ, k)
        resultados.append(ResultadoTeste(
            familia="chi2_janelas",
            teste=f"concursos {inicio + 1}-{inicio + largura}",
            estatistica=x2, p_valor=p, efeito=w,
            detalhe={"largura": largura},
        ))
    return resultados


def ks_uniforme(valores: np.ndarray, minimo: int, n_universo: int,
                seed: int) -> ResultadoTeste:
    """KS de uniformidade sobre todas as dezenas sorteadas (agrupadas).

    KS exige distribuição contínua; aplicamos a transformação de jitter
    u = (d - min + U[0,1)) / N, exatamente Uniform(0,1) sob H0 (Pearson).
    Seed fixa para reprodutibilidade.
    """
    rng = np.random.default_rng(seed)
    u = (valores - minimo + rng.random(valores.size)) / n_universo
    est, p = stats.kstest(u, "uniform")
    return ResultadoTeste(
        familia="ks_uniformidade", teste="dezenas agrupadas",
        estatistica=float(est), p_valor=float(p), efeito=float(est),
        detalhe={"n": int(valores.size), "jitter_seed": seed},
    )


def entropia_g(indicadores: np.ndarray, k: int) -> ResultadoTeste:
    """Entropia de Shannon da distribuição por dezena + teste G (razão de
    verossimilhança), com o mesmo ajuste de Joe do qui-quadrado."""
    m, n_univ = indicadores.shape
    contagens = indicadores.sum(axis=0).astype(float)
    total = contagens.sum()
    freq = contagens / total
    nz = freq > 0
    h = float(-(freq[nz] * np.log2(freq[nz])).sum())
    h_max = np.log2(n_univ)
    # correção de viés de Miller-Madow
    h_mm = h + (nz.sum() - 1) / (2 * total * np.log(2))
    esperado = total / n_univ
    g = 2.0 * float((contagens[nz] * np.log(contagens[nz] / esperado)).sum())
    g_aj = g * (n_univ - 1) / (n_univ - k)
    p = float(stats.chi2.sf(g_aj, df=n_univ - 1))
    return ResultadoTeste(
        familia="entropia", teste="distribuição por dezena (teste G)",
        estatistica=g_aj, p_valor=p,
        efeito=float(h_max - h_mm),
        detalhe={"entropia_bits": h_mm, "maxima_bits": h_max},
    )
