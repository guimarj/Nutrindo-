"""Subconjunto da bateria NIST SP 800-22 sobre o bitstream derivado.

Implementados (fórmulas da revisão 1a do SP 800-22): Frequency (Monobit),
Block Frequency, Runs, Longest Run of Ones, DFT Spectral, Cumulative Sums,
Approximate Entropy e Serial. Testes que exigem streams muito maiores que o
histórico das loterias produz (Maurer ~387k bits, Linear Complexity, Random
Excursions) são omitidos com registro do motivo.

Inclui ainda o teste de compressibilidade (não-NIST): o stream real é
comprimido (zlib nível 9) e comparado à distribuição de tamanhos comprimidos
de streams aleatórios de mesmo comprimento — p-valor de Monte Carlo.
"""

import math
import zlib

import numpy as np
from scipy import special, stats

from .resultado import ResultadoTeste

_FAMILIA = "nist_sp800_22"


def bateria_nist(bits: np.ndarray) -> list[ResultadoTeste]:
    n = len(bits)
    resultados = []
    if n < 100:
        return [ResultadoTeste(
            familia=_FAMILIA, teste="(bateria não executada)",
            estatistica=float("nan"), p_valor=1.0,
            detalhe={"motivo": f"stream com {n} bits < 100"})]

    resultados.append(_monobit(bits))
    resultados.append(_block_frequency(bits, M=128))
    resultados.append(_runs(bits))
    resultados.append(_longest_run(bits))
    resultados.append(_dft(bits))
    resultados.extend(_cusum(bits))
    resultados.append(_approx_entropy(bits, m=2))
    resultados.extend(_serial(bits, m=3))
    return [r for r in resultados if r is not None]


def _monobit(bits) -> ResultadoTeste:
    n = len(bits)
    s = float(np.abs(2.0 * bits.sum() - n)) / math.sqrt(n)
    p = float(special.erfc(s / math.sqrt(2)))
    return ResultadoTeste(_FAMILIA, "frequency (monobit)", s, p,
                          efeito=float(bits.mean() - 0.5), detalhe={"n": n})


def _block_frequency(bits, M=128) -> ResultadoTeste | None:
    n = len(bits)
    nb = n // M
    if nb < 20:
        return None
    props = bits[:nb * M].reshape(nb, M).mean(axis=1)
    x2 = float(4.0 * M * ((props - 0.5) ** 2).sum())
    p = float(special.gammaincc(nb / 2.0, x2 / 2.0))
    return ResultadoTeste(_FAMILIA, f"block frequency (M={M})", x2, p,
                          detalhe={"blocos": nb})


def _runs(bits) -> ResultadoTeste:
    n = len(bits)
    pi = float(bits.mean())
    if abs(pi - 0.5) >= 2.0 / math.sqrt(n):
        return ResultadoTeste(_FAMILIA, "runs", float("nan"), 0.0,
                              detalhe={"motivo": "pré-requisito monobit falhou"})
    v = 1 + int((bits[1:] != bits[:-1]).sum())
    num = abs(v - 2.0 * n * pi * (1 - pi))
    den = 2.0 * math.sqrt(2.0 * n) * pi * (1 - pi)
    p = float(special.erfc(num / den))
    return ResultadoTeste(_FAMILIA, "runs", float(v), p, detalhe={"pi": pi})


def _longest_run(bits) -> ResultadoTeste | None:
    n = len(bits)
    if n >= 6272:
        M, K = 128, 5
        cats = [4, 5, 6, 7, 8, 9]          # ≤4, 5, 6, 7, 8, ≥9
        pis = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
    elif n >= 128:
        M, K = 8, 3
        cats = [1, 2, 3, 4]                # ≤1, 2, 3, ≥4
        pis = [0.2148, 0.3672, 0.2305, 0.1875]
    else:
        return None
    nb = n // M
    blocos = bits[:nb * M].reshape(nb, M)
    contagens = np.zeros(K + 1)
    for bloco in blocos:
        maior = _maior_run(bloco)
        idx = 0
        while idx < K and maior > cats[idx]:
            idx += 1
        contagens[idx] += 1
    esperados = nb * np.array(pis)
    x2 = float(((contagens - esperados) ** 2 / esperados).sum())
    p = float(special.gammaincc(K / 2.0, x2 / 2.0))
    return ResultadoTeste(_FAMILIA, f"longest run of ones (M={M})", x2, p,
                          detalhe={"blocos": nb})


def _maior_run(bloco: np.ndarray) -> int:
    maior = atual = 0
    for b in bloco:
        atual = atual + 1 if b else 0
        maior = max(maior, atual)
    return maior


def _dft(bits) -> ResultadoTeste:
    n = len(bits)
    x = 2.0 * bits.astype(float) - 1.0
    mod = np.abs(np.fft.rfft(x))[: n // 2]
    limiar = math.sqrt(math.log(1.0 / 0.05) * n)
    n0 = 0.95 * n / 2.0
    n1 = float((mod < limiar).sum())
    d = (n1 - n0) / math.sqrt(n * 0.95 * 0.05 / 4.0)
    p = float(special.erfc(abs(d) / math.sqrt(2)))
    return ResultadoTeste(_FAMILIA, "dft spectral", d, p,
                          detalhe={"picos_acima": int(n // 2 - n1)})


def _cusum(bits) -> list[ResultadoTeste]:
    n = len(bits)
    x = 2 * bits.astype(np.int64) - 1
    resultados = []
    for modo, serie in (("forward", x), ("backward", x[::-1])):
        z = float(np.abs(np.cumsum(serie)).max())
        p = _cusum_p(n, z)
        resultados.append(ResultadoTeste(
            _FAMILIA, f"cumulative sums ({modo})", z, p, detalhe={}))
    return resultados


def _cusum_p(n: int, z: float) -> float:
    raiz = math.sqrt(n)
    soma1 = sum(
        stats.norm.cdf((4 * k + 1) * z / raiz) - stats.norm.cdf((4 * k - 1) * z / raiz)
        for k in range(int((-n / z + 1) // 4), int((n / z - 1) // 4) + 1)
    )
    soma2 = sum(
        stats.norm.cdf((4 * k + 3) * z / raiz) - stats.norm.cdf((4 * k + 1) * z / raiz)
        for k in range(int((-n / z - 3) // 4), int((n / z - 1) // 4) + 1)
    )
    return float(min(max(1.0 - soma1 + soma2, 0.0), 1.0))


def _phi(bits: np.ndarray, m: int) -> float:
    if m == 0:
        return 0.0
    n = len(bits)
    estendido = np.concatenate([bits, bits[: m - 1]])
    # índice inteiro de cada janela de m bits
    pesos = 1 << np.arange(m - 1, -1, -1)
    idx = np.lib.stride_tricks.sliding_window_view(estendido, m) @ pesos
    idx = idx[:n].astype(np.int64)
    contagens = np.bincount(idx, minlength=1 << m).astype(float)
    freq = contagens[contagens > 0] / n
    return float((freq * np.log(freq)).sum())


def _approx_entropy(bits, m=2) -> ResultadoTeste:
    n = len(bits)
    apen = _phi(bits, m) - _phi(bits, m + 1)
    x2 = 2.0 * n * (math.log(2) - apen)
    p = float(special.gammaincc(2 ** (m - 1), x2 / 2.0))
    return ResultadoTeste(_FAMILIA, f"approximate entropy (m={m})", x2, p,
                          detalhe={"apen": apen})


def _psi2(bits: np.ndarray, m: int) -> float:
    if m <= 0:
        return 0.0
    n = len(bits)
    estendido = np.concatenate([bits, bits[: m - 1]])
    pesos = 1 << np.arange(m - 1, -1, -1)
    idx = np.lib.stride_tricks.sliding_window_view(estendido, m) @ pesos
    idx = idx[:n].astype(np.int64)
    contagens = np.bincount(idx, minlength=1 << m).astype(float)
    return float((2 ** m / n) * (contagens ** 2).sum() - n)


def _serial(bits, m=3) -> list[ResultadoTeste]:
    d1 = _psi2(bits, m) - _psi2(bits, m - 1)
    d2 = _psi2(bits, m) - 2 * _psi2(bits, m - 1) + _psi2(bits, m - 2)
    p1 = float(special.gammaincc(2 ** (m - 2), d1 / 2.0))
    p2 = float(special.gammaincc(2 ** (m - 3), d2 / 2.0))
    return [
        ResultadoTeste(_FAMILIA, f"serial ∇ψ² (m={m})", d1, p1, detalhe={}),
        ResultadoTeste(_FAMILIA, f"serial ∇²ψ² (m={m})", d2, p2, detalhe={}),
    ]


def compressibilidade(bits: np.ndarray, n_simulacoes: int = 300,
                      seed: int = 20260806) -> ResultadoTeste:
    """p-valor de Monte Carlo: fração de streams aleatórios que comprimem
    tão bem ou melhor que o real (unicaudal — estrutura => compressão)."""
    real = len(zlib.compress(np.packbits(bits).tobytes(), 9))
    rng = np.random.default_rng(seed)
    sint = np.array([
        len(zlib.compress(np.packbits(rng.integers(0, 2, len(bits),
                                                   dtype=np.uint8)).tobytes(), 9))
        for _ in range(n_simulacoes)
    ])
    p = float((1 + (sint <= real).sum()) / (n_simulacoes + 1))
    return ResultadoTeste(
        familia="compressibilidade", teste="zlib-9 vs streams aleatórios",
        estatistica=float(real), p_valor=p,
        efeito=float(real / sint.mean() - 1),
        detalhe={"bytes_real": real, "bytes_medio_aleatorio": float(sint.mean()),
                 "n_simulacoes": n_simulacoes},
    )
