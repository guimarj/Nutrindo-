"""CLI da suíte. Fase 1 expõe `atualizar` e `status`; as fases seguintes
adicionam `analisar`, `backtest` e `palpites` como novos subcomandos."""

import argparse
import sys
from pathlib import Path

from . import db, ingestao
from .config import JOGOS


def _jogos_do_argumento(valor: str) -> list[str]:
    if valor == "todos":
        return list(JOGOS)
    if valor not in JOGOS:
        raise SystemExit(f"jogo inválido: {valor!r}; opções: todos, {', '.join(JOGOS)}")
    return [valor]


def _log(*args, file=None, **kwargs):
    print(*args, file=file or sys.stdout, **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="loteria",
        description="Suíte de análise estatística das loterias da Caixa",
    )
    parser.add_argument("--db", type=Path, default=db.DB_PADRAO,
                        help=f"caminho do SQLite (padrão: {db.DB_PADRAO})")
    sub = parser.add_subparsers(dest="comando", required=True)

    p_atu = sub.add_parser("atualizar", help="baixa/atualiza o histórico")
    p_atu.add_argument("--jogo", default="todos",
                       help="megasena, lotofacil, quina, supersete ou todos")
    p_atu.add_argument("--fonte", choices=["auto", "caixa", "espelho"],
                       default="auto",
                       help="auto: tenta a API da Caixa e cai para o espelho GitHub")
    p_atu.add_argument("--sem-enriquecer", action="store_true",
                       help="não busca metadados de concursos vindos do espelho")

    sub.add_parser("status", help="resumo do banco: concursos, lacunas, metadados")

    args = parser.parse_args(argv)
    con = db.conectar(args.db)
    try:
        if args.comando == "atualizar":
            ingestao.atualizar(con, _jogos_do_argumento(args.jogo),
                               fonte=args.fonte,
                               enriquecer=not args.sem_enriquecer, log=_log)
        elif args.comando == "status":
            ingestao.status(con, log=_log)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
