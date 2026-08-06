"""Fase 5 — Modelagem de popularidade: quanta gente tende a jogar uma
combinação, e portanto quanto o prêmio seria dividido se ela saísse.

Modelo log-linear: densidade_de_apostas(c) ∝ exp(Σ βi·fi(c)), onde fi são
padrões que a população comprovadamente joga (datas ≤31, sequências,
múltiplos, simetrias no volante, dezenas quentes, repetição do último
resultado). O `multiplicador(c)` é normalizado por Monte Carlo para que a
média sobre combinações uniformes seja 1: multiplicador 3 significa "espere
3× mais apostas iguais à sua do que numa combinação típica".

Duas origens para os βi:
1. AJUSTE AOS DADOS: regressão de Poisson (IRLS) do nº de ganhadores de uma
   faixa intermediária por concurso sobre as features da combinação sorteada,
   com offset de volume (mediana móvel de ganhadores). Requer os metadados da
   API oficial da Caixa no banco.
2. FALLBACK DOCUMENTADO: pesos heurísticos calibrados qualitativamente à
   literatura de loteria (Simon 1998; Farrell & Walker 1999; Riedwyl 2002 —
   viés de datas/aniversários, sequências e padrões geométricos são os
   efeitos mais fortes e replicados). Usado quando o banco só tem dezenas.

O relatório sempre informa qual origem está em uso.
"""

import math
from dataclasses import dataclass

import numpy as np

from .config import Jogo

FAIXA_AJUSTE = {"megasena": 4, "lotofacil": 12, "quina": 3, "supersete": 4}

# β padrão (fallback). Sinal positivo = padrão mais jogado que a média.
BETAS_PADRAO = {
    "frac_baixas": 1.2,     # fração de dezenas <= 31 (datas), centrada
    "todas_baixas": 0.9,    # combinação inteira "jogável como datas"
    "seq_pares": 1.5,       # pares consecutivos (1-2, 33-34...)
    "progressao": 1.0,      # progressão aritmética completa (1-2-3..., 5-10-15...)
    "soma_baixa": 0.5,      # soma abaixo do típico (datas puxam para baixo)
    "concentracao_volante": 0.8,  # dezenas amontoadas numa linha/coluna
    "quentes_recentes": 0.6,      # dezenas mais sorteadas nos últimos concursos
    "repete_anterior": 0.4,       # dezenas do último resultado
}
BETAS_PADRAO_POSICIONAL = {
    "digito_repetido": 1.0,  # 7-7-7-7-7-7-7 e afins
    "sequencia": 0.8,        # 1-2-3-4-5-6-7
}

_NOMES_FEATURES = list(BETAS_PADRAO)
_NOMES_FEATURES_POS = list(BETAS_PADRAO_POSICIONAL)


@dataclass
class ModeloPopularidade:
    jogo: str
    betas: dict[str, float]
    origem: str              # 'ajustado (Poisson/IRLS)' | 'fallback (literatura)'
    normalizacao: float      # E[exp(eta)] sob combinações uniformes
    contexto: dict           # último sorteio, dezenas quentes etc.

    def eta(self, dezenas: list[int], cfg: Jogo) -> tuple[float, dict[str, float]]:
        f = extrair_features(dezenas, cfg, self.contexto)
        eta = sum(self.betas.get(nome, 0.0) * v for nome, v in f.items())
        return eta, f

    def multiplicador(self, dezenas: list[int], cfg: Jogo) -> float:
        """>1: mais disputada que a média; <1: menos disputada."""
        eta, _ = self.eta(dezenas, cfg)
        return math.exp(eta) / self.normalizacao


# ------------------------------------------------------------------ features

def extrair_features(dezenas: list[int], cfg: Jogo,
                     contexto: dict) -> dict[str, float]:
    if cfg.posicional:
        return _features_posicional(dezenas)
    k = len(dezenas)
    ordenadas = sorted(dezenas)
    n_baixas = sum(1 for d in ordenadas if d <= 31)
    esperado_baixas = k * min(31, cfg.universo_max) / cfg.tamanho_universo

    difs = [b - a for a, b in zip(ordenadas, ordenadas[1:])]
    seq_pares = sum(1 for d in difs if d == 1)
    progressao = 1.0 if len(set(difs)) == 1 else 0.0

    soma = sum(ordenadas)
    mu = k * (cfg.universo_min + cfg.universo_max) / 2
    sigma = math.sqrt(k * (cfg.tamanho_universo ** 2 - 1) / 12
                      * (cfg.tamanho_universo - k) / (cfg.tamanho_universo - 1))
    z_soma = (soma - mu) / sigma

    colunas = 5 if cfg.tamanho_universo == 25 else 10
    linhas = [0] * ((cfg.universo_max // colunas) + 1)
    cols = [0] * colunas
    for d in ordenadas:
        linhas[(d - cfg.universo_min) // colunas] += 1
        cols[(d - cfg.universo_min) % colunas] += 1
    concentracao = max(max(linhas), max(cols)) / k

    quentes = contexto.get("quentes", set())
    anterior = contexto.get("anterior", set())
    return {
        "frac_baixas": (n_baixas - esperado_baixas) / k,
        "todas_baixas": 1.0 if n_baixas == k and cfg.universo_max > 31 else 0.0,
        "seq_pares": seq_pares / (k - 1),
        "progressao": progressao,
        "soma_baixa": max(0.0, -z_soma) / 2,
        "concentracao_volante": max(0.0, concentracao - 2 / math.sqrt(k)),
        "quentes_recentes": (len(quentes & set(ordenadas)) / k) if quentes else 0.0,
        "repete_anterior": (len(anterior & set(ordenadas)) / k) if anterior else 0.0,
    }


def _features_posicional(digitos: list[int]) -> dict[str, float]:
    contagem = max(digitos.count(d) for d in set(digitos))
    difs = [b - a for a, b in zip(digitos, digitos[1:])]
    return {
        "digito_repetido": max(0.0, (contagem - 2) / (len(digitos) - 2)),
        "sequencia": 1.0 if len(set(difs)) == 1 and difs[0] in (-1, 0, 1) else 0.0,
    }


# ------------------------------------------------------------------- montagem

def montar_modelo(sorteios, cfg: Jogo, seed: int = 20260806,
                  janela_quentes: int = 50) -> ModeloPopularidade:
    """Ajusta aos dados se houver metadados de ganhadores; senão, fallback."""
    contexto = _contexto(sorteios, cfg, janela_quentes)
    nomes = _NOMES_FEATURES_POS if cfg.posicional else _NOMES_FEATURES
    padrao = BETAS_PADRAO_POSICIONAL if cfg.posicional else BETAS_PADRAO

    betas, origem = None, "fallback (literatura)"
    ajuste = _ajustar_poisson(sorteios, cfg, contexto, nomes)
    if ajuste is not None:
        betas = dict(zip(nomes, ajuste))
        origem = "ajustado (Poisson/IRLS sobre ganhadores por concurso)"
    else:
        betas = dict(padrao)

    normalizacao = _normalizar(betas, cfg, contexto, seed)
    return ModeloPopularidade(jogo=cfg.slug, betas=betas, origem=origem,
                              normalizacao=normalizacao, contexto=contexto)


def _contexto(sorteios, cfg, janela_quentes) -> dict:
    if not sorteios:
        return {"quentes": set(), "anterior": set()}
    recentes = sorteios[-janela_quentes:]
    freq: dict[int, int] = {}
    for s in recentes:
        for d in s.dezenas:
            freq[d] = freq.get(d, 0) + 1
    n_quentes = max(5, cfg.dezenas_sorteadas)
    quentes = set(sorted(freq, key=freq.get, reverse=True)[:n_quentes])
    return {"quentes": quentes, "anterior": set(sorteios[-1].dezenas)}


def _normalizar(betas, cfg, contexto, seed, amostras: int = 20000) -> float:
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(amostras):
        if cfg.posicional:
            dez = rng.integers(0, 10, cfg.dezenas_sorteadas).tolist()
        else:
            dez = sorted((rng.choice(cfg.tamanho_universo, cfg.dezenas_sorteadas,
                                     replace=False) + cfg.universo_min).tolist())
        f = extrair_features([int(d) for d in dez], cfg, contexto)
        total += math.exp(sum(betas.get(n, 0.0) * v for n, v in f.items()))
    return total / amostras


# ------------------------------------------------- ajuste Poisson (com dados)

def _ajustar_poisson(sorteios, cfg, contexto, nomes) -> np.ndarray | None:
    """ganhadores_faixa_t ~ Poisson(volume_t · exp(Xβ)). Retorna β ou None."""
    faixa_alvo = FAIXA_AJUSTE.get(cfg.slug)
    ys, linhas = [], []
    for s in sorteios:
        if not s.premios:
            continue
        ganhadores = _ganhadores_da_faixa(s.premios, faixa_alvo)
        if ganhadores is None:
            continue
        f = extrair_features(s.dezenas, cfg, contexto)
        linhas.append([f[n] for n in nomes])
        ys.append(ganhadores)
    if len(ys) < 300:   # metadados insuficientes para um ajuste sério
        return None
    y = np.asarray(ys, dtype=float)
    X = np.asarray(linhas, dtype=float)
    # offset de volume: mediana móvel dos ganhadores (janela 21) captura a
    # variação de arrecadação (acúmulos atraem mais apostas)
    volume = _mediana_movel(y, 21)
    volume[volume < 1] = 1.0
    return irls_poisson(X, y, np.log(volume))


def _ganhadores_da_faixa(premios: list[dict], acertos: int) -> int | None:
    for p in premios:
        desc = str(p.get("descricao", "")).lower()
        if str(acertos) in desc.split() or f"{acertos} acertos" in desc:
            return p.get("ganhadores")
    # fallback: ordena por faixa numérica quando a descrição não ajuda
    for p in premios:
        if p.get("faixa") == acertos:
            return p.get("ganhadores")
    return None


def _mediana_movel(y: np.ndarray, janela: int) -> np.ndarray:
    meia = janela // 2
    out = np.empty_like(y)
    for i in range(len(y)):
        ini, fim = max(0, i - meia), min(len(y), i + meia + 1)
        out[i] = np.median(y[ini:fim])
    return out


def irls_poisson(X: np.ndarray, y: np.ndarray, offset: np.ndarray,
                 iteracoes: int = 50, tol: float = 1e-8) -> np.ndarray:
    """Regressão de Poisson com intercepto e offset, por IRLS."""
    Xa = np.column_stack([np.ones(len(y)), X])
    beta = np.zeros(Xa.shape[1])
    for _ in range(iteracoes):
        eta = offset + Xa @ beta
        mu = np.exp(np.clip(eta, -30, 30))
        z = eta - offset + (y - mu) / mu
        W = mu
        XtW = Xa.T * W
        beta_novo = np.linalg.solve(XtW @ Xa + 1e-9 * np.eye(Xa.shape[1]),
                                    XtW @ z)
        if np.max(np.abs(beta_novo - beta)) < tol:
            beta = beta_novo
            break
        beta = beta_novo
    return beta[1:]  # descarta o intercepto (absorvido pela normalização)
