"""Fase 4 — Backtest com split temporal estrito.

Toda estratégia é uma função que enxerga SOMENTE os concursos de treino
(1..N) e produz uma aposta fixa; a aposta é avaliada nos concursos N+1 em
diante. O desempenho é comparado à distribuição do mesmo métrico sobre
`simulacoes` apostas uniformemente aleatórias avaliadas no MESMO período de
teste, com intervalo central de 95%.

Resultado dentro do intervalo = indistinguível de sorte. O relatório trata
resultado negativo com o mesmo destaque do positivo — essa é a regra do
projeto.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import db
from .config import Jogo, jogo as _jogo

PASTA_RELATORIOS = Path("relatorios")


@dataclass
class ResultadoEstrategia:
    estrategia: str
    aposta: list[int]
    media_acertos: float
    media_acertos_aleatoria: float
    ic95_media: tuple[float, float]
    percentil: float                  # posição vs apostas aleatórias (0-100)
    acertos_por_faixa: dict[str, int] = field(default_factory=dict)
    veredito: str = ""


# ---------------------------------------------------------------- estratégias

def _quentes(ind_treino: np.ndarray, cfg: Jogo) -> np.ndarray:
    """Dezenas mais frequentes no treino (explora o desvio de frequência que
    as Fases 2-3 confirmaram — se ele for preditivo, esta estratégia vence)."""
    freq = ind_treino.sum(axis=0)
    return np.sort(np.argsort(-freq, kind="stable")[:cfg.dezenas_sorteadas])


def _frias(ind_treino: np.ndarray, cfg: Jogo) -> np.ndarray:
    """Dezenas menos frequentes ('estão atrasadas' — falácia do jogador)."""
    freq = ind_treino.sum(axis=0)
    return np.sort(np.argsort(freq, kind="stable")[:cfg.dezenas_sorteadas])


def _trinca_quente(ind_treino: np.ndarray, cfg: Jogo) -> np.ndarray:
    """Trinca mais coocorrente do treino + dezenas mais quentes no resto."""
    x = ind_treino.astype(np.int32)
    co = x.T @ x
    n = co.shape[0]
    melhor, melhor_v = None, -1
    freq = ind_treino.sum(axis=0)
    for i in range(n):
        for j in range(i + 1, n):
            for l in range(j + 1, n):
                v = min(co[i, j], co[i, l], co[j, l])
                if v > melhor_v:
                    melhor_v, melhor = v, (i, j, l)
    base = list(melhor)
    resto = [d for d in np.argsort(-freq, kind="stable") if d not in base]
    return np.sort(np.array(base + resto[:cfg.dezenas_sorteadas - 3]))


ESTRATEGIAS = {
    "dezenas_quentes": _quentes,
    "dezenas_frias": _frias,
    "trinca_quente": _trinca_quente,
}


def _quentes_posicional(digitos_treino: np.ndarray) -> np.ndarray:
    return np.array([np.bincount(digitos_treino[:, c], minlength=10).argmax()
                     for c in range(digitos_treino.shape[1])])


def _frias_posicional(digitos_treino: np.ndarray) -> np.ndarray:
    return np.array([np.bincount(digitos_treino[:, c], minlength=10).argmin()
                     for c in range(digitos_treino.shape[1])])


ESTRATEGIAS_POSICIONAIS = {
    "digitos_quentes": _quentes_posicional,
    "digitos_frios": _frias_posicional,
}


# ---------------------------------------------------------------- avaliação

def _avaliar_apostas(apostas_ind: np.ndarray, ind_teste: np.ndarray) -> np.ndarray:
    """apostas_ind (A × N) × ind_teste (m × N) -> acertos (A × m)."""
    return apostas_ind.astype(np.int32) @ ind_teste.T.astype(np.int32)


def backtest(con, slug: str, corte: float = 0.7, simulacoes: int = 10_000,
             seed: int = 20260806, salvar_json: bool = True,
             log=print) -> list[ResultadoEstrategia]:
    cfg = _jogo(slug)
    sorteios = db.carregar(con, slug)
    if len(sorteios) < 100:
        log(f"histórico insuficiente para {cfg.nome}")
        return []
    n_treino = int(len(sorteios) * corte)
    treino, teste = sorteios[:n_treino], sorteios[n_treino:]
    log(f"[{cfg.nome}] treino: concursos 1-{n_treino}; teste: "
        f"{n_treino + 1}-{len(sorteios)} ({len(teste)} concursos); "
        f"{simulacoes} apostas aleatórias de referência")

    rng = np.random.default_rng(seed)
    if cfg.posicional:
        resultados = _backtest_posicional(treino, teste, cfg, simulacoes, rng)
    else:
        resultados = _backtest_combinatorio(treino, teste, cfg, simulacoes, rng)

    log(f"\n{'estratégia':18s} {'aposta':32s} {'média':>7s} "
        f"{'aleat.':>7s} {'IC95% aleatório':>17s} {'perc.':>6s}  veredito")
    log("-" * 110)
    for r in resultados:
        aposta_str = " ".join(f"{d:02d}" for d in r.aposta)
        log(f"{r.estrategia:18s} {aposta_str:32s} {r.media_acertos:7.4f} "
            f"{r.media_acertos_aleatoria:7.4f} "
            f"[{r.ic95_media[0]:6.4f}, {r.ic95_media[1]:6.4f}] "
            f"{r.percentil:5.1f}%  {r.veredito}")

    fora = [r for r in resultados if "FORA" in r.veredito]
    if not fora:
        log("\nRESULTADO NEGATIVO (reportado com o mesmo destaque de um "
            "positivo): nenhuma estratégia baseada no histórico supera "
            "apostas aleatórias fora da amostra. Os desvios das Fases 2-3 "
            "NÃO são preditivos.")
    else:
        log("\nAtenção: estratégia(s) fora do IC de 95%. Antes de concluir "
            "vantagem, considere o número de estratégias testadas "
            "(multiplicidade) e refaça com outro corte temporal.")

    if salvar_json:
        PASTA_RELATORIOS.mkdir(exist_ok=True)
        destino = PASTA_RELATORIOS / f"backtest_{slug}.json"
        with open(destino, "w", encoding="utf-8") as f:
            json.dump({"jogo": slug, "corte": corte, "n_treino": n_treino,
                       "n_teste": len(teste), "simulacoes": simulacoes,
                       "resultados": [asdict(r) for r in resultados]},
                      f, ensure_ascii=False, indent=1)
        log(f"detalhamento salvo em {destino}")
    return resultados


def _indicadores(sorteios, cfg) -> np.ndarray:
    ind = np.zeros((len(sorteios), cfg.tamanho_universo), dtype=np.uint8)
    for t, s in enumerate(sorteios):
        for d in s.dezenas:
            ind[t, d - cfg.universo_min] = 1
    return ind


def _backtest_combinatorio(treino, teste, cfg, simulacoes, rng):
    n_univ, k = cfg.tamanho_universo, cfg.dezenas_sorteadas
    ind_treino = _indicadores(treino, cfg)
    ind_teste = _indicadores(teste, cfg)

    # referência: apostas uniformes
    aleatorias = np.zeros((simulacoes, n_univ), dtype=np.uint8)
    for a in range(simulacoes):
        aleatorias[a, rng.choice(n_univ, k, replace=False)] = 1
    acertos_alea = _avaliar_apostas(aleatorias, ind_teste)  # (S × m)
    medias_alea = acertos_alea.mean(axis=1)
    ic = (float(np.quantile(medias_alea, 0.025)),
          float(np.quantile(medias_alea, 0.975)))

    resultados = []
    for nome, estrategia in ESTRATEGIAS.items():
        idx = estrategia(ind_treino, cfg)
        aposta_ind = np.zeros(n_univ, dtype=np.uint8)
        aposta_ind[idx] = 1
        acertos = (ind_teste @ aposta_ind).astype(int)
        resultados.append(_montar(nome, idx + cfg.universo_min, acertos,
                                  medias_alea, ic, acertos_alea, cfg))
    return resultados


def _backtest_posicional(treino, teste, cfg, simulacoes, rng):
    dig_treino = np.array([s.dezenas for s in treino])
    dig_teste = np.array([s.dezenas for s in teste])
    n_col = dig_treino.shape[1]

    aleatorias = rng.integers(0, 10, size=(simulacoes, n_col))
    acertos_alea = (aleatorias[:, None, :] == dig_teste[None, :, :]).sum(axis=2)
    medias_alea = acertos_alea.mean(axis=1)
    ic = (float(np.quantile(medias_alea, 0.025)),
          float(np.quantile(medias_alea, 0.975)))

    resultados = []
    for nome, estrategia in ESTRATEGIAS_POSICIONAIS.items():
        aposta = estrategia(dig_treino)
        acertos = (dig_teste == aposta[None, :]).sum(axis=1)
        resultados.append(_montar(nome, aposta, acertos, medias_alea, ic,
                                  acertos_alea, cfg))
    return resultados


def _montar(nome, aposta, acertos, medias_alea, ic, acertos_alea, cfg):
    media = float(acertos.mean())
    percentil = float(
        (np.count_nonzero(medias_alea < media)
         + 0.5 * np.count_nonzero(medias_alea == media)) / len(medias_alea) * 100)
    dentro = ic[0] <= media <= ic[1]
    faixas = {f"{f}+ acertos": int((acertos >= f).sum())
              for f in cfg.faixas_premiadas}
    return ResultadoEstrategia(
        estrategia=nome, aposta=[int(d) for d in aposta],
        media_acertos=media,
        media_acertos_aleatoria=float(medias_alea.mean()),
        ic95_media=ic, percentil=percentil,
        acertos_por_faixa=faixas,
        veredito=("dentro do esperado por acaso" if dentro
                  else "FORA do IC95% — investigar"),
    )
