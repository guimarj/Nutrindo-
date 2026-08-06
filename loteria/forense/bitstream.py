"""Derivação de bitstream sem viés a partir dos sorteios.

Codificar cada dezena em binário direto enviesaria o stream (ex.: 6 bits
representam 0-63 mas a Mega-Sena só usa 1-60). Em vez disso, cada concurso é
convertido no seu POSTO COMBINATÓRIO: a combinação sorteada vira um inteiro
uniforme em [0, C(N,k)) (ordem colexicográfica; posto = Σ C(v_i, i+1) com
valores 0-based ordenados). Como C(N,k) não é potência de 2, aplicamos
rejeição: se posto < 2^b com b = ⌊log2 C(N,k)⌋, emitimos b bits; senão o
concurso é descartado. Sob H0 o resultado são bits iid Bernoulli(1/2) exatos.

Super Sete: posto = Σ dígito_i · 10^i, uniforme em [0, 10^7), mesma rejeição.

Limitação documentada: o posto usa a combinação ordenada, então a ordem de
saída das bolas dentro do concurso não entra no bitstream (ela é testada
separadamente por Ljung-Box/runs nas séries indicadoras).
"""

from math import comb

import numpy as np


def posto_combinacao(dezenas_1based: list[int]) -> int:
    """Posto colexicográfico da combinação (valores 1-based)."""
    v = sorted(d - 1 for d in dezenas_1based)
    return sum(comb(val, i + 1) for i, val in enumerate(v))


def posto_supersete(digitos: list[int]) -> int:
    return sum(d * 10 ** i for i, d in enumerate(digitos))


def derivar_bits(sorteios: list[list[int]], n_universo: int, k: int,
                 posicional: bool = False) -> tuple[np.ndarray, float]:
    """Retorna (array de bits 0/1 uint8, taxa de aproveitamento dos concursos)."""
    if posicional:
        maximo = 10 ** len(sorteios[0])
        postos = [posto_supersete(s) for s in sorteios]
    else:
        maximo = comb(n_universo, k)
        postos = [posto_combinacao(s) for s in sorteios]

    b = maximo.bit_length() - 1  # ⌊log2 maximo⌋
    limite = 1 << b
    bits: list[np.ndarray] = []
    aceitos = 0
    for posto in postos:
        if posto < limite:
            aceitos += 1
            bits.append(np.array([(posto >> i) & 1 for i in range(b - 1, -1, -1)],
                                 dtype=np.uint8))
    stream = np.concatenate(bits) if bits else np.zeros(0, dtype=np.uint8)
    return stream, aceitos / len(postos) if postos else 0.0


def taxa_aceitacao_teorica(n_universo: int, k: int, posicional: bool,
                           n_colunas: int = 7) -> float:
    maximo = 10 ** n_colunas if posicional else comb(n_universo, k)
    return (1 << (maximo.bit_length() - 1)) / maximo
