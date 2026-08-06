"""Comando `analisar`: roda a bateria forense sobre um jogo, imprime o
resumo por família e salva o detalhamento em JSON para o relatório HTML."""

import json
from dataclasses import asdict
from pathlib import Path

from . import db, forense
from .config import jogo as _jogo

PASTA_RELATORIOS = Path("relatorios")


def analisar(con, slug: str, janela: int = 250, passo: int = 125,
             incluir_trincas: bool = True, alfa: float = 0.05,
             salvar_json: bool = True, log=print):
    cfg = _jogo(slug)
    sorteios = db.carregar(con, slug)
    if not sorteios:
        log(f"sem dados para {cfg.nome}; rode `atualizar` antes")
        return None

    log(f"[{cfg.nome}] bateria forense sobre {len(sorteios)} concursos...")
    resultados = forense.rodar_bateria(sorteios, cfg, janela=janela,
                                       passo=passo,
                                       incluir_trincas=incluir_trincas)
    resumos = forense.resumir(resultados, alfa=alfa)

    log(f"\n{'família':28s} {'testes':>7s} {'min p':>10s} {'min q':>10s}  veredito")
    log("-" * 92)
    for r in sorted(resumos, key=lambda r: r.q_minimo):
        log(f"{r.familia:28s} {r.n_testes:7d} {r.p_minimo:10.4g} "
            f"{r.q_minimo:10.4g}  {r.veredito}"
            + (f"  [{r.destaque}]" if "REJEITA" in r.veredito else ""))

    achados = [r for r in resultados if r.q_valor is not None and r.q_valor < alfa]
    if achados:
        log(f"\n{len(achados)} teste(s) individuais com q < {alfa}:")
        for r in sorted(achados, key=lambda r: r.q_valor)[:20]:
            log(f"  {r.familia} / {r.teste}: p={r.p_valor:.3g} q={r.q_valor:.3g} "
                f"efeito={r.efeito if r.efeito is None else round(r.efeito, 4)}")
        log("Atenção: um achado só vira evidência se sobreviver ao controle "
            "negativo da Fase 3 (mesma bateria em dados sintéticos).")
    else:
        log(f"\nNenhuma família rejeita H0 após correção BH (alfa={alfa}). "
            f"Compatível com sorteios uniformes e independentes.")

    if salvar_json:
        PASTA_RELATORIOS.mkdir(exist_ok=True)
        destino = PASTA_RELATORIOS / f"forense_{slug}.json"
        with open(destino, "w", encoding="utf-8") as f:
            json.dump({
                "jogo": slug, "concursos": len(sorteios), "alfa": alfa,
                "parametros": {"janela": janela, "passo": passo,
                               "trincas": incluir_trincas,
                               "seed_jitter": forense.SEED_JITTER},
                "resumos": [asdict(r) for r in resumos],
                "resultados": [asdict(r) for r in resultados],
            }, f, ensure_ascii=False, indent=1, default=str)
        log(f"detalhamento salvo em {destino}")
    return resultados
