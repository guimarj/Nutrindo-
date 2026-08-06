"""Bateria forense: executa os 9 grupos de testes sobre um jogo e aplica a
correção de múltiplas comparações por família.

Uso programático:
    resultados = rodar_bateria(sorteios, cfg)
    resumos = resumir(resultados)

A mesma função roda sobre dados reais (Fase 2) e sintéticos (Fase 3) — é isso
que torna o controle negativo comparável.
"""

import numpy as np

from ..config import Jogo
from ..db import Sorteio
from . import bitstream, coocorrencia, frequencia, nist, temporal
from .resultado import ResultadoTeste, ResumoFamilia, corrigir_bh, resumir

__all__ = ["rodar_bateria", "resumir", "corrigir_bh",
           "ResultadoTeste", "ResumoFamilia"]

SEED_JITTER = 20260806  # reprodutibilidade do KS; documentado no relatório


def rodar_bateria(sorteios: list[Sorteio], cfg: Jogo,
                  janela: int = 250, passo: int = 125,
                  incluir_trincas: bool = True) -> list[ResultadoTeste]:
    listas = [s.dezenas for s in sorteios]
    if cfg.posicional:
        resultados = _bateria_posicional(listas, cfg)
    else:
        resultados = _bateria_combinatoria(listas, cfg, janela, passo,
                                           incluir_trincas)

    bits, aproveitamento = bitstream.derivar_bits(
        listas, cfg.tamanho_universo, cfg.dezenas_sorteadas,
        posicional=cfg.posicional)
    nist_res = nist.bateria_nist(bits)
    for r in nist_res:
        r.detalhe.setdefault("bits", int(len(bits)))
        r.detalhe.setdefault("aproveitamento", round(aproveitamento, 3))
    resultados.extend(nist_res)
    resultados.append(nist.compressibilidade(bits))

    return corrigir_bh(resultados)


def _bateria_combinatoria(listas, cfg, janela, passo, incluir_trincas):
    m = len(listas)
    n_univ = cfg.tamanho_universo
    k = cfg.dezenas_sorteadas

    indicadores = np.zeros((m, n_univ), dtype=np.uint8)
    for t, dezenas in enumerate(listas):
        for d in dezenas:
            indicadores[t, d - cfg.universo_min] = 1
    valores = np.array([d for dezenas in listas for d in dezenas])
    series = {f"dezena {i + cfg.universo_min:02d}": indicadores[:, i]
              for i in range(n_univ)}
    soma_por_concurso = np.array([sum(d) for d in listas], dtype=float)
    series_soma = {"soma dos concursos (acima/abaixo da mediana)":
                   (soma_por_concurso > np.median(soma_por_concurso)).astype(np.uint8)}

    resultados = [
        frequencia.chi2_global(indicadores, k),
        *frequencia.chi2_janelas(indicadores, k, janela, passo),
        frequencia.ks_uniforme(valores, cfg.universo_min, n_univ, SEED_JITTER),
        frequencia.entropia_g(indicadores, k),
        *temporal.ljung_box(series),
        *temporal.runs_wald_wolfowitz({**series, **series_soma}),
        *temporal.espectral_fisher(series),
        *temporal.gaps_geometrico(series, p_sucesso=k / n_univ),
        *coocorrencia.pares(indicadores, k),
    ]
    if incluir_trincas:
        idx = [np.array([d - cfg.universo_min for d in dezenas])
               for dezenas in listas]
        resultados.extend(coocorrencia.trincas(idx, n_univ, m, k))
    return resultados


def _bateria_posicional(listas, cfg):
    """Super Sete: cada coluna é um sorteio independente de 1 dígito em 0-9."""
    digitos = np.array(listas, dtype=np.int64)  # (m, 7)
    m, n_col = digitos.shape

    resultados = []
    series: dict[str, np.ndarray] = {}
    for c in range(n_col):
        ind_col = np.zeros((m, 10), dtype=np.uint8)
        ind_col[np.arange(m), digitos[:, c]] = 1
        r = frequencia.chi2_global(ind_col, k=1)
        r.teste = f"coluna {c + 1}"
        resultados.append(r)
        resultados.extend(_renomear(
            frequencia.chi2_janelas(ind_col, 1, largura=200, passo=100),
            prefixo=f"coluna {c + 1}, "))
        for dig in range(10):
            series[f"coluna {c + 1} dígito {dig}"] = ind_col[:, dig]

    valores = digitos.flatten()
    resultados.append(frequencia.ks_uniforme(valores, 0, 10, SEED_JITTER))
    ind_geral = np.zeros((m * n_col, 10), dtype=np.uint8)
    ind_geral[np.arange(m * n_col), valores] = 1
    resultados.append(frequencia.entropia_g(ind_geral, k=1))

    resultados.extend(temporal.ljung_box(series))
    resultados.extend(temporal.runs_wald_wolfowitz(series))
    resultados.extend(temporal.espectral_fisher(series))
    resultados.extend(temporal.gaps_geometrico(series, p_sucesso=0.1))
    resultados.extend(coocorrencia.independencia_colunas(digitos))
    return resultados


def _renomear(resultados, prefixo):
    for r in resultados:
        r.teste = prefixo + r.teste
    return resultados
