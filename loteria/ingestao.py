"""Orquestração da ingestão: escolhe fonte, atualiza incrementalmente,
enriquece metadados quando a fonte oficial está alcançável."""

import sqlite3
import sys

from . import db
from .config import JOGOS
from .sources import FonteCaixa, FonteEspelhoGithub


def atualizar(con: sqlite3.Connection, jogos: list[str],
              fonte: str = "auto", enriquecer: bool = True,
              log=print) -> None:
    """fonte: 'auto' tenta caixa e cai para o espelho; ou força 'caixa'/'espelho'."""
    caixa = FonteCaixa()
    espelho = FonteEspelhoGithub()

    if fonte == "caixa":
        principal = caixa
    elif fonte == "espelho":
        principal = espelho
    else:
        if caixa.disponivel():
            principal = caixa
        elif espelho.disponivel():
            principal = espelho
            log("aviso: API da Caixa inalcançável; usando espelho GitHub "
                "(sem metadados de premiação — enriqueça depois com "
                "`atualizar --fonte caixa`)")
        else:
            log("erro: nenhuma fonte alcançável", file=sys.stderr)
            return

    for slug in jogos:
        ultimo = db.ultimo_concurso(con, slug)
        log(f"[{JOGOS[slug].nome}] último concurso no banco: {ultimo or '—'}; "
            f"buscando novos via {principal.nome}...")
        novos = principal.buscar(slug, apos_concurso=ultimo)
        res = db.gravar(con, novos)
        log(f"  +{res.inseridos} inseridos, {res.enriquecidos} enriquecidos, "
            f"{res.ignorados} já existiam")
        for msg in res.invalidos:
            log(f"  INVÁLIDO (rejeitado): {msg}", file=sys.stderr)

        faltando = db.lacunas(con, slug)
        if faltando:
            log(f"  atenção: {len(faltando)} lacunas na série "
                f"(ex.: {faltando[:10]})")

        if enriquecer and principal is caixa:
            pendentes = db.concursos_sem_metadados(con, slug)
            if pendentes:
                log(f"  enriquecendo {len(pendentes)} concursos sem metadados...")
                lote = [s for n in pendentes
                        if (s := caixa.buscar_um(slug, n)) is not None]
                res = db.gravar(con, lote)
                log(f"  {res.enriquecidos} enriquecidos")


def status(con: sqlite3.Connection, log=print) -> None:
    for slug, cfg in JOGOS.items():
        ultimo = db.ultimo_concurso(con, slug)
        total = con.execute(
            "SELECT COUNT(*) AS c FROM concursos WHERE jogo = ?", (slug,)
        ).fetchone()["c"]
        com_meta = con.execute(
            "SELECT COUNT(*) AS c FROM concursos WHERE jogo = ? AND fonte = 'caixa'",
            (slug,)
        ).fetchone()["c"]
        n_lacunas = len(db.lacunas(con, slug))
        log(f"{cfg.nome:12s} concursos: {total:6d}  último: {ultimo:6d}  "
            f"com metadados: {com_meta:6d}  lacunas: {n_lacunas}")
