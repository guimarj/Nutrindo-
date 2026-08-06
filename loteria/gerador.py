"""Fase 6 — Gerador de palpites.

Score multicritério transparente sobre um pool de candidatos uniformes:

  a) anti_rateio  — minimizar disputa esperada (Fase 5). É o critério de
     MAIOR peso: não muda a chance de ganhar, muda quanto se leva se ganhar.
  b) aderencia    — soma no intervalo interquartil histórico, par/ímpar,
     faixas do volante e linhas/colunas compatíveis com o que sai.
  c) diversificacao — sobreposição mínima entre os N jogos da carteira
     (seleção gulosa com penalidade de interseção).
  d) desvio       — SOMENTE se as Fases 2-3 confirmaram desvio de frequência
     (arquivo relatorios/controle_<jogo>.json) o score ganha um componente
     extra com peso proporcional ao tamanho de efeito (limitado a 0.15).
     O achado de janela histórica da Quina não entra: um desvio de regime
     antigo não diz nada sobre o próximo concurso.

REGRA DE INTEGRIDADE: a probabilidade de acerto exibida é a real e idêntica
para qualquer aposta; o gerador otimiza apenas o valor condicional do prêmio.
"""

import json
import math
from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

from . import db, popularidade
from .config import Jogo, jogo as _jogo

PASTA_RELATORIOS = Path("relatorios")

RODAPE = ("Otimizado para valor do prêmio, não para chance de acerto. "
          "A probabilidade é a mesma de qualquer outro jogo.")

# volume típico de apostas simples-equivalentes e prêmio principal típico;
# grosseiros por definição — ajuste com --volume/--premio
VOLUME_PADRAO = {"megasena": 30e6, "lotofacil": 25e6, "quina": 15e6,
                 "supersete": 2e6}
PREMIO_PADRAO = {"megasena": 50e6, "lotofacil": 1.8e6, "quina": 10e6,
                 "supersete": 2.5e6}

PESOS_POR_MODO = {
    "padrao":      {"anti_rateio": 0.45, "aderencia": 0.30, "diversificacao": 0.25},
    "conservador": {"anti_rateio": 0.25, "aderencia": 0.55, "diversificacao": 0.20},
    "contrarian":  {"anti_rateio": 0.65, "aderencia": 0.10, "diversificacao": 0.25},
}
PESO_DESVIO_MAXIMO = 0.15


@dataclass
class Palpite:
    dezenas: list[int]
    scores: dict[str, float]
    score_final: float
    justificativa: str
    probabilidade: str
    multiplicador_disputa: float
    premio_condicional: float


@dataclass
class Carteira:
    jogo: str
    modo: str
    palpites: list[Palpite]
    custo_total: float
    pesos: dict[str, float]
    origem_popularidade: str
    desvio_incorporado: str | None
    avisos: list[str] = field(default_factory=list)
    rodape: str = RODAPE


# ------------------------------------------------------------ desvio (6d)

def _carregar_desvio(slug: str, sorteios, cfg) -> tuple[np.ndarray | None, str | None, float]:
    """z-scores de frequência por dezena, se e só se o controle negativo
    confirmou desvio de frequência global para este jogo."""
    caminho = PASTA_RELATORIOS / f"controle_{slug}.json"
    if not caminho.exists():
        return None, None, 0.0
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    confirmadas = {c["familia"] for c in dados["comparacoes"]
                   if c["veredito"].startswith("ACHADO")}
    if not confirmadas & {"chi2_frequencia", "entropia"}:
        return None, None, 0.0

    m = len(sorteios)
    contagens = np.zeros(cfg.tamanho_universo)
    for s in sorteios:
        for d in s.dezenas:
            contagens[d - cfg.universo_min] += 1
    p = cfg.dezenas_sorteadas / cfg.tamanho_universo
    esperado, sd = m * p, math.sqrt(m * p * (1 - p))
    z = (contagens - esperado) / sd

    efeito = _efeito_chi2(slug)
    peso = min(PESO_DESVIO_MAXIMO, 2.0 * efeito)
    origem = (f"desvio de frequência confirmado pelo controle negativo "
              f"(efeito w={efeito:.3f}; peso {peso:.2f})")
    return z, origem, peso


def _efeito_chi2(slug: str) -> float:
    caminho = PASTA_RELATORIOS / f"forense_{slug}.json"
    if caminho.exists():
        dados = json.loads(caminho.read_text(encoding="utf-8"))
        for r in dados["resultados"]:
            if r["familia"] == "chi2_frequencia" and r["teste"] == "global":
                return float(r["efeito"] or 0.0)
    return 0.0


# ------------------------------------------------------------ aderência

class _PerfilHistorico:
    def __init__(self, sorteios, cfg: Jogo):
        somas = np.array([sum(s.dezenas) for s in sorteios], dtype=float)
        self.q1, self.q3 = np.quantile(somas, [0.25, 0.75])
        self.iqr = max(self.q3 - self.q1, 1.0)

        k = cfg.dezenas_sorteadas
        pares = np.array([sum(1 for d in s.dezenas if d % 2 == 0)
                          for s in sorteios])
        self.prob_pares = np.bincount(pares, minlength=k + 1) / len(sorteios)

        self.colunas = 5 if cfg.tamanho_universo == 25 else 10
        self.n_linhas = (cfg.universo_max - cfg.universo_min) // self.colunas + 1
        linhas_hist = np.zeros(self.n_linhas)
        for s in sorteios:
            for d in s.dezenas:
                linhas_hist[(d - cfg.universo_min) // self.colunas] += 1
        self.frac_linhas = linhas_hist / linhas_hist.sum()
        self.cfg = cfg

    def score(self, dezenas: list[int]) -> float:
        cfg = self.cfg
        soma = sum(dezenas)
        if self.q1 <= soma <= self.q3:
            s_soma = 1.0
        else:
            dist = min(abs(soma - self.q1), abs(soma - self.q3))
            s_soma = max(0.0, 1.0 - dist / self.iqr)

        n_pares = sum(1 for d in dezenas if d % 2 == 0)
        s_par = float(self.prob_pares[n_pares] / self.prob_pares.max())

        linhas = np.zeros(self.n_linhas)
        for d in dezenas:
            linhas[(d - cfg.universo_min) // self.colunas] += 1
        tv = 0.5 * np.abs(linhas / len(dezenas) - self.frac_linhas).sum()
        s_linhas = float(max(0.0, 1.0 - 2.0 * tv))
        return (s_soma + s_par + s_linhas) / 3.0


# ------------------------------------------------------------ geração

def gerar_carteira(con, slug: str, n: int = 10, modo: str = "padrao",
                   orcamento: float | None = None,
                   volume: float | None = None, premio: float | None = None,
                   pool: int = 4000, seed: int | None = None,
                   universo_fechamento: int = 12) -> Carteira:
    cfg = _jogo(slug)
    sorteios = db.carregar(con, slug)
    if not sorteios:
        raise SystemExit(f"sem dados para {cfg.nome}; rode `atualizar` antes")
    avisos = []

    if orcamento is not None:
        n_orc = int(orcamento // cfg.preco_aposta_simples)
        if n_orc < 1:
            raise SystemExit(
                f"orçamento R${orcamento:.2f} não cobre uma aposta simples "
                f"(R${cfg.preco_aposta_simples:.2f})")
        avisos.append(f"orçamento R${orcamento:.2f} => {n_orc} apostas simples "
                      f"(sobra R${orcamento - n_orc * cfg.preco_aposta_simples:.2f}); "
                      f"confira o preço vigente na Caixa")
        n = n_orc

    modelo = popularidade.montar_modelo(sorteios, cfg)
    volume = volume or VOLUME_PADRAO[slug]
    premio = premio or PREMIO_PADRAO[slug]

    if modo == "fechamento":
        return _carteira_fechamento(cfg, sorteios, modelo, n, volume, premio,
                                    universo_fechamento, avisos, orcamento)

    pesos = dict(PESOS_POR_MODO[modo])
    z_desvio, origem_desvio, peso_desvio = (None, None, 0.0)
    if modo != "conservador":
        z_desvio, origem_desvio, peso_desvio = _carregar_desvio(slug, sorteios, cfg)
    if peso_desvio > 0:
        pesos = {k: v * (1 - peso_desvio) for k, v in pesos.items()}
        pesos["desvio"] = peso_desvio

    rng = np.random.default_rng(seed)
    candidatos = _amostrar_candidatos(cfg, pool, rng)
    perfil = _PerfilHistorico(sorteios, cfg)

    # componentes por candidato
    mults = np.array([modelo.multiplicador(c, cfg) for c in candidatos])
    anti = _percentil_invertido(mults)          # menos disputado => 1
    ader = np.array([perfil.score(c) for c in candidatos]) \
        if not cfg.posicional else np.full(len(candidatos), 0.5)
    desv = np.zeros(len(candidatos))
    if z_desvio is not None:
        desv = np.array([
            math.tanh(np.mean([z_desvio[d - cfg.universo_min] for d in c]))
            for c in candidatos]) * 0.5 + 0.5

    base = (pesos["anti_rateio"] * anti + pesos["aderencia"] * ader
            + pesos.get("desvio", 0.0) * desv)

    escolhidos = _selecao_gulosa(candidatos, base, n, cfg,
                                 pesos["diversificacao"])

    palpites = []
    for idx, s_div in escolhidos:
        c = candidatos[idx]
        scores = {"anti_rateio": round(float(anti[idx]), 3),
                  "aderencia": round(float(ader[idx]), 3),
                  "diversificacao": round(float(s_div), 3)}
        if peso_desvio > 0:
            scores["desvio"] = round(float(desv[idx]), 3)
        final = sum(pesos[k] * scores[k] for k in pesos)
        palpites.append(Palpite(
            dezenas=list(c), scores=scores, score_final=round(float(final), 3),
            justificativa=_justificar(c, scores, mults[idx], cfg),
            probabilidade=_probabilidade(cfg),
            multiplicador_disputa=round(float(mults[idx]), 2),
            premio_condicional=_premio_condicional(premio, volume, mults[idx], cfg),
        ))

    return Carteira(jogo=slug, modo=modo, palpites=palpites,
                    custo_total=round(n * cfg.preco_aposta_simples, 2),
                    pesos={k: round(v, 3) for k, v in pesos.items()},
                    origem_popularidade=modelo.origem,
                    desvio_incorporado=origem_desvio, avisos=avisos)


def _amostrar_candidatos(cfg, pool, rng):
    vistos, candidatos = set(), []
    while len(candidatos) < pool:
        if cfg.posicional:
            c = tuple(int(x) for x in rng.integers(0, 10, cfg.dezenas_sorteadas))
        else:
            c = tuple(sorted(int(x) + cfg.universo_min for x in
                             rng.choice(cfg.tamanho_universo,
                                        cfg.dezenas_sorteadas, replace=False)))
        if c not in vistos:
            vistos.add(c)
            candidatos.append(c)
    return candidatos


def _percentil_invertido(valores: np.ndarray) -> np.ndarray:
    ordem = valores.argsort().argsort()          # rank: menor valor => rank 0
    return 1.0 - ordem / (len(valores) - 1)


def _sobreposicao(a, b, posicional: bool) -> int:
    if posicional:
        return sum(1 for x, y in zip(a, b) if x == y)
    return len(set(a) & set(b))


def _selecao_gulosa(candidatos, base, n, cfg, peso_div):
    k = cfg.dezenas_sorteadas
    escolhidos: list[tuple[int, float]] = []
    usados: list = []
    disponiveis = set(range(len(candidatos)))
    while len(escolhidos) < n and disponiveis:
        melhor, melhor_v, melhor_div = None, -1e9, 1.0
        for i in disponiveis:
            if usados:
                overlap = max(_sobreposicao(candidatos[i], u, cfg.posicional)
                              for u in usados)
                s_div = 1.0 - overlap / k
            else:
                s_div = 1.0
            v = base[i] + peso_div * s_div
            if v > melhor_v:
                melhor, melhor_v, melhor_div = i, v, s_div
        escolhidos.append((melhor, melhor_div))
        usados.append(candidatos[melhor])
        disponiveis.remove(melhor)
    return escolhidos


def _probabilidade(cfg: Jogo) -> str:
    total = (10 ** cfg.dezenas_sorteadas if cfg.posicional
             else comb(cfg.tamanho_universo, cfg.dezenas_sorteadas))
    return f"1 em {total:,}".replace(",", ".")


def _premio_condicional(premio, volume, mult, cfg) -> float:
    total = (10 ** cfg.dezenas_sorteadas if cfg.posicional
             else comb(cfg.tamanho_universo, cfg.dezenas_sorteadas))
    co_ganhadores = volume * mult / total
    return round(premio / (1.0 + co_ganhadores), 2)


def _justificar(c, scores, mult, cfg) -> str:
    partes = []
    if mult < 0.8:
        partes.append(f"disputa {mult:.2f}× a média (poucos padrões populares)")
    elif mult > 1.5:
        partes.append(f"atenção: disputa {mult:.2f}× a média")
    if scores.get("aderencia", 0) >= 0.7:
        partes.append("perfil de soma/paridade/faixas típico do histórico")
    if scores.get("diversificacao", 0) >= 0.8:
        partes.append("baixa sobreposição com os demais jogos da carteira")
    if scores.get("desvio") is not None and scores["desvio"] > 0.55:
        partes.append("leve inclinação às dezenas do desvio confirmado")
    return "; ".join(partes) or "melhor equilíbrio dos critérios no pool"


# ------------------------------------------------------------ fechamento

def fechamento_quadra(universo: list[int], k_aposta: int) -> list[list[int]]:
    """Cobertura gulosa: se 5 dezenas sorteadas ⊂ universo, algum jogo tem
    >=4 delas (garantia de quadra). Verificada exaustivamente ao final."""
    quintetos = list(combinations(sorted(universo), 5))
    jogos_candidatos = list(combinations(sorted(universo), k_aposta))
    pendentes = set(range(len(quintetos)))
    cobre: list[set[int]] = []
    for jc in jogos_candidatos:
        sjc = set(jc)
        cobre.append({qi for qi in range(len(quintetos))
                      if len(sjc & set(quintetos[qi])) >= 4})
    escolhidos = []
    while pendentes:
        melhor = max(range(len(jogos_candidatos)),
                     key=lambda j: len(cobre[j] & pendentes))
        ganho = cobre[melhor] & pendentes
        if not ganho:
            raise RuntimeError("cobertura impossível — universo pequeno demais")
        escolhidos.append(list(jogos_candidatos[melhor]))
        pendentes -= ganho
    assert verificar_fechamento(universo, escolhidos), "garantia violada"
    return escolhidos


def verificar_fechamento(universo, jogos) -> bool:
    """Checagem exaustiva e independente da garantia de quadra."""
    for quinteto in combinations(sorted(universo), 5):
        sq = set(quinteto)
        if not any(len(sq & set(j)) >= 4 for j in jogos):
            return False
    return True


def _carteira_fechamento(cfg, sorteios, modelo, n, volume, premio,
                         tamanho_universo, avisos, orcamento):
    if cfg.slug not in ("megasena", "quina"):
        raise SystemExit("--modo fechamento disponível para megasena e quina")

    # universo: dezenas individualmente menos populares (altas, frias no
    # sentido de popularidade, fora do último resultado)
    scores_dezena = []
    anterior = set(sorteios[-1].dezenas)
    quentes = modelo.contexto.get("quentes", set())
    for d in cfg.universo:
        s = 0.0
        s += 0.6 if d > 31 else 0.0
        s += 0.2 if d not in quentes else 0.0
        s += 0.2 if d not in anterior else 0.0
        scores_dezena.append((s, d))
    universo = sorted(d for _, d in
                      sorted(scores_dezena, reverse=True)[:tamanho_universo])

    jogos = fechamento_quadra(universo, cfg.dezenas_sorteadas)
    custo = len(jogos) * cfg.preco_aposta_simples
    if orcamento is not None and custo > orcamento:
        avisos.append(f"fechamento de {len(jogos)} jogos custa "
                      f"R${custo:.2f} > orçamento; reduza --universo")

    palpites = []
    for j in jogos:
        mult = modelo.multiplicador(j, cfg)
        palpites.append(Palpite(
            dezenas=j, scores={"cobertura": 1.0},
            score_final=1.0,
            justificativa=(f"parte do fechamento: quadra garantida se 5 das "
                           f"{len(universo)} dezenas do universo saírem"),
            probabilidade=_probabilidade(cfg),
            multiplicador_disputa=round(float(mult), 2),
            premio_condicional=_premio_condicional(premio, volume, mult, cfg),
        ))
    avisos.append(f"universo do fechamento: "
                  + " ".join(f"{d:02d}" for d in universo)
                  + f" | {len(jogos)} jogos, garantia verificada exaustivamente")
    return Carteira(jogo=cfg.slug, modo="fechamento", palpites=palpites,
                    custo_total=round(custo, 2), pesos={},
                    origem_popularidade=modelo.origem,
                    desvio_incorporado=None, avisos=avisos)


# ------------------------------------------------------------ saída

def salvar_carteira(carteira: Carteira) -> Path:
    PASTA_RELATORIOS.mkdir(exist_ok=True)
    destino = PASTA_RELATORIOS / f"palpites_{carteira.jogo}.json"
    from dataclasses import asdict
    with open(destino, "w", encoding="utf-8") as f:
        json.dump(asdict(carteira), f, ensure_ascii=False, indent=1)
    return destino


def imprimir_carteira(carteira: Carteira, cfg: Jogo, log=print):
    log(f"\n=== {cfg.nome} — {len(carteira.palpites)} jogos, modo "
        f"{carteira.modo}, custo estimado R${carteira.custo_total:.2f} ===")
    log(f"popularidade: {carteira.origem_popularidade}")
    if carteira.desvio_incorporado:
        log(f"componente de desvio ATIVO: {carteira.desvio_incorporado}")
    if carteira.pesos:
        log("pesos: " + ", ".join(f"{k}={v}" for k, v in carteira.pesos.items()))
    for a in carteira.avisos:
        log(f"aviso: {a}")
    for i, p in enumerate(carteira.palpites, 1):
        dez = " ".join(f"{d:02d}" for d in p.dezenas)
        log(f"\n#{i:02d}  {dez}")
        log(f"     scores: " + ", ".join(f"{k}={v}" for k, v in p.scores.items())
            + f" | final={p.score_final}")
        log(f"     {p.justificativa}")
        log(f"     probabilidade real: {p.probabilidade} | disputa "
            f"{p.multiplicador_disputa}× | prêmio esperado se acertar: "
            f"{_reais(p.premio_condicional)}")
    log(f"\n{RODAPE}")


def _reais(v: float) -> str:
    return ("R$" + f"{v:,.2f}"
            .replace(",", "X").replace(".", ",").replace("X", "."))
