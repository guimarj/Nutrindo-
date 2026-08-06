"""Fase 3 — Controle negativo.

Gera históricos sintéticos do mesmo tamanho do real usando RNG CRIPTOGRÁFICO
(secrets.SystemRandom, que lê do pool de entropia do sistema operacional) e
roda a MESMA bateria forense em cada réplica.

Regra de decisão, por família de testes:
  - Se o q mínimo real >= alfa: família limpa, nada a explicar.
  - Se o q mínimo real < alfa: calculamos o p-valor empírico
        p_emp = (1 + nº de réplicas com q_min <= q_min_real) / (R + 1)
    e a taxa de falso alarme (fração de réplicas que também rejeitam a alfa).
    O achado só é CONFIRMADO se p_emp <= alfa — ou seja, se históricos
    genuinamente aleatórios quase nunca produzem uma rejeição tão forte.
    Como p_emp >= 1/(R+1), confirmar a alfa=0.05 exige R >= 19 réplicas.

Isso implementa a regra do projeto: um desvio só conta se aparecer nos dados
reais E não aparecer no controle sintético.
"""

import json
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path

from . import db, forense
from .config import Jogo, jogo as _jogo
from .db import Sorteio

PASTA_RELATORIOS = Path("relatorios")

VEREDITO_LIMPO = "limpo (real não rejeita)"
VEREDITO_CONFIRMADO = "ACHADO CONFIRMADO (real rejeita; sintético não)"
VEREDITO_FALSO = "falso positivo provável (sintético também rejeita)"


@dataclass
class ComparacaoFamilia:
    familia: str
    q_min_real: float
    q_min_sintetico_mediana: float
    taxa_rejeicao_sintetica: float   # fração de réplicas com q_min < alfa
    p_empirico: float | None         # só quando o real rejeita
    veredito: str
    destaque_real: str


def gerar_historia_sintetica(cfg: Jogo, m: int) -> list[Sorteio]:
    rng = secrets.SystemRandom()
    sorteios = []
    for i in range(m):
        if cfg.posicional:
            dezenas = [rng.randrange(10) for _ in range(cfg.dezenas_sorteadas)]
        else:
            dezenas = sorted(rng.sample(list(cfg.universo), cfg.dezenas_sorteadas))
        sorteios.append(Sorteio(jogo=cfg.slug, concurso=i + 1,
                                dezenas=dezenas, fonte="caixa"))
    return sorteios


def _qmin_por_familia(resultados) -> dict[str, tuple[float, str]]:
    melhor: dict[str, tuple[float, str]] = {}
    for r in resultados:
        q = r.q_valor if r.q_valor is not None else 1.0
        if r.familia not in melhor or q < melhor[r.familia][0]:
            melhor[r.familia] = (q, r.teste)
    return melhor


def controle_negativo(con, slug: str, replicas: int = 20, alfa: float = 0.05,
                      janela: int = 250, passo: int = 125,
                      incluir_trincas: bool = True, salvar_json: bool = True,
                      log=print) -> list[ComparacaoFamilia]:
    cfg = _jogo(slug)
    reais = db.carregar(con, slug)
    if not reais:
        log(f"sem dados para {cfg.nome}; rode `atualizar` antes")
        return []

    minimo_replicas = int(1 / alfa) - 1   # p_emp mínimo = 1/(R+1) <= alfa
    if replicas < minimo_replicas:
        log(f"aviso: com {replicas} réplicas o menor p empírico possível é "
            f"1/{replicas + 1} > alfa={alfa}; nenhum achado poderá ser "
            f"confirmado. Use --replicas >= {minimo_replicas}.")

    log(f"[{cfg.nome}] bateria nos dados reais ({len(reais)} concursos)...")
    res_real = forense.rodar_bateria(reais, cfg, janela=janela, passo=passo,
                                     incluir_trincas=incluir_trincas)
    real = _qmin_por_familia(res_real)

    log(f"[{cfg.nome}] {replicas} réplicas sintéticas (RNG criptográfico)...")
    sinteticos: list[dict[str, tuple[float, str]]] = []
    for r in range(replicas):
        historia = gerar_historia_sintetica(cfg, len(reais))
        res = forense.rodar_bateria(historia, cfg, janela=janela, passo=passo,
                                    incluir_trincas=incluir_trincas)
        sinteticos.append(_qmin_por_familia(res))
        if (r + 1) % 5 == 0:
            log(f"  réplica {r + 1}/{replicas}")

    comparacoes = []
    for familia in sorted(real):
        q_real, destaque = real[familia]
        qs_sint = sorted(s.get(familia, (1.0, ""))[0] for s in sinteticos)
        mediana = qs_sint[len(qs_sint) // 2]
        taxa = sum(q < alfa for q in qs_sint) / replicas
        if q_real >= alfa:
            veredito, p_emp = VEREDITO_LIMPO, None
        else:
            p_emp = (1 + sum(q <= q_real for q in qs_sint)) / (replicas + 1)
            veredito = (VEREDITO_CONFIRMADO if p_emp <= alfa
                        else VEREDITO_FALSO)
        comparacoes.append(ComparacaoFamilia(
            familia=familia, q_min_real=float(q_real),
            q_min_sintetico_mediana=float(mediana),
            taxa_rejeicao_sintetica=taxa, p_empirico=p_emp,
            veredito=veredito, destaque_real=destaque,
        ))

    log(f"\n{'família':24s} {'q real':>9s} {'q sint (med)':>13s} "
        f"{'rej sint':>9s} {'p emp':>8s}  veredito")
    log("-" * 100)
    for c in sorted(comparacoes, key=lambda c: c.q_min_real):
        p_emp = "—" if c.p_empirico is None else f"{c.p_empirico:.3f}"
        log(f"{c.familia:24s} {c.q_min_real:9.3g} "
            f"{c.q_min_sintetico_mediana:13.3g} "
            f"{c.taxa_rejeicao_sintetica:9.0%} {p_emp:>8s}  {c.veredito}"
            + (f"  [{c.destaque_real}]"
               if c.veredito == VEREDITO_CONFIRMADO else ""))

    confirmados = [c for c in comparacoes if c.veredito == VEREDITO_CONFIRMADO]
    if confirmados:
        log(f"\n{len(confirmados)} achado(s) sobrevivem ao controle negativo. "
            "Interpretação exige cautela: desvio estatístico ≠ desvio "
            "explorável; confira também qualidade da fonte de dados.")
    else:
        log("\nNenhum achado sobrevive ao controle negativo: o comportamento "
            "dos dados reais é indistinguível de sorteios criptograficamente "
            "aleatórios sob esta bateria.")

    if salvar_json:
        PASTA_RELATORIOS.mkdir(exist_ok=True)
        destino = PASTA_RELATORIOS / f"controle_{slug}.json"
        with open(destino, "w", encoding="utf-8") as f:
            json.dump({
                "jogo": slug, "replicas": replicas, "alfa": alfa,
                "concursos": len(reais),
                "comparacoes": [asdict(c) for c in comparacoes],
                "q_min_sintetico_por_replica": [
                    {fam: q for fam, (q, _) in s.items()} for s in sinteticos],
            }, f, ensure_ascii=False, indent=1)
        log(f"detalhamento salvo em {destino}")
    return comparacoes
