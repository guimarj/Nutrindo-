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

    p_ana = sub.add_parser("analisar", help="bateria forense de aleatoriedade")
    p_ana.add_argument("--jogo", default="todos")
    p_ana.add_argument("--janela", type=int, default=250,
                       help="largura das janelas móveis do qui-quadrado")
    p_ana.add_argument("--passo", type=int, default=125)
    p_ana.add_argument("--sem-trincas", action="store_true",
                       help="pula a análise de trincas (mais rápida)")
    p_ana.add_argument("--alfa", type=float, default=0.05,
                       help="limiar de FDR (Benjamini-Hochberg)")

    p_ctl = sub.add_parser("controle",
                           help="controle negativo: mesma bateria em réplicas "
                                "sintéticas de RNG criptográfico")
    p_ctl.add_argument("--jogo", default="todos")
    p_ctl.add_argument("--replicas", type=int, default=20)
    p_ctl.add_argument("--janela", type=int, default=250)
    p_ctl.add_argument("--passo", type=int, default=125)
    p_ctl.add_argument("--sem-trincas", action="store_true")
    p_ctl.add_argument("--alfa", type=float, default=0.05)

    p_bt = sub.add_parser("backtest",
                          help="estratégias históricas vs apostas aleatórias, "
                               "com split temporal estrito")
    p_bt.add_argument("--jogo", default="todos")
    p_bt.add_argument("--corte", type=float, default=0.7,
                      help="fração do histórico usada como treino (padrão 0.7)")
    p_bt.add_argument("--simulacoes", type=int, default=10_000)

    args = parser.parse_args(argv)
    con = db.conectar(args.db)
    try:
        if args.comando == "atualizar":
            ingestao.atualizar(con, _jogos_do_argumento(args.jogo),
                               fonte=args.fonte,
                               enriquecer=not args.sem_enriquecer, log=_log)
        elif args.comando == "status":
            ingestao.status(con, log=_log)
        elif args.comando == "analisar":
            from . import analise
            for slug in _jogos_do_argumento(args.jogo):
                analise.analisar(con, slug, janela=args.janela,
                                 passo=args.passo,
                                 incluir_trincas=not args.sem_trincas,
                                 alfa=args.alfa, log=_log)
                _log("")
        elif args.comando == "controle":
            from . import controle
            for slug in _jogos_do_argumento(args.jogo):
                controle.controle_negativo(
                    con, slug, replicas=args.replicas, janela=args.janela,
                    passo=args.passo, incluir_trincas=not args.sem_trincas,
                    alfa=args.alfa, log=_log)
                _log("")
        elif args.comando == "backtest":
            from . import backtest
            for slug in _jogos_do_argumento(args.jogo):
                backtest.backtest(con, slug, corte=args.corte,
                                  simulacoes=args.simulacoes, log=_log)
                _log("")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
