"""Formato uniforme de resultado e correção para múltiplas comparações.

Regra de integridade da suíte: nenhum p-valor é reportado sem q-valor
Benjamini-Hochberg calculado dentro da sua família de testes. O veredito de
uma família usa o menor q, nunca o menor p.
"""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ResultadoTeste:
    familia: str            # ex.: 'chi2_frequencia', 'coocorrencia_pares'
    teste: str              # ex.: 'global', 'dezena 07', 'janela 500-750'
    estatistica: float
    p_valor: float
    efeito: float | None = None   # tamanho de efeito na escala natural do teste
    detalhe: dict = field(default_factory=dict)
    q_valor: float | None = None  # preenchido por corrigir_bh


def bh_qvalores(p: np.ndarray) -> np.ndarray:
    """q-valores de Benjamini-Hochberg (step-up)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    ordem = np.argsort(p)
    q = np.empty(n)
    minimo = 1.0
    for pos in range(n - 1, -1, -1):
        i = ordem[pos]
        minimo = min(minimo, p[i] * n / (pos + 1))
        q[i] = minimo
    return q


def corrigir_bh(resultados: list[ResultadoTeste]) -> list[ResultadoTeste]:
    """Aplica BH dentro de cada família, in loco.

    Famílias cujos membros já chegam com q_valor preenchido (ex.:
    coocorrência, que corrige sobre o vetor completo antes de truncar o
    relatório ao top-N) são preservadas como estão.
    """
    familias: dict[str, list[ResultadoTeste]] = defaultdict(list)
    for r in resultados:
        familias[r.familia].append(r)
    for grupo in familias.values():
        if all(r.q_valor is not None for r in grupo):
            continue
        qs = bh_qvalores([r.p_valor for r in grupo])
        for r, q in zip(grupo, qs):
            r.q_valor = float(q)
    return resultados


@dataclass
class ResumoFamilia:
    familia: str
    n_testes: int
    p_minimo: float
    q_minimo: float
    veredito: str           # 'compatível com aleatoriedade' | 'REJEITA H0 (q<alfa)'
    destaque: str           # teste com menor q


def resumir(resultados: list[ResultadoTeste], alfa: float = 0.05) -> list[ResumoFamilia]:
    familias: dict[str, list[ResultadoTeste]] = defaultdict(list)
    for r in resultados:
        familias[r.familia].append(r)
    resumos = []
    for nome, grupo in familias.items():
        melhor = min(grupo, key=lambda r: (r.q_valor, r.p_valor))
        rejeita = melhor.q_valor is not None and melhor.q_valor < alfa
        resumos.append(ResumoFamilia(
            familia=nome, n_testes=len(grupo),
            p_minimo=min(r.p_valor for r in grupo),
            q_minimo=melhor.q_valor,
            veredito="REJEITA H0 (q<%.2f)" % alfa if rejeita
                     else "compatível com aleatoriedade",
            destaque=melhor.teste,
        ))
    return resumos
