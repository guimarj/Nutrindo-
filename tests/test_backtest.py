"""Backtest: calibração (dados uniformes -> estratégias dentro do IC) e
poder (vício forte e persistente -> 'quentes' vence fora da amostra)."""

import unittest

import numpy as np

from loteria import backtest, db
from loteria.controle import gerar_historia_sintetica
from loteria.config import JOGOS
from loteria.db import Sorteio


def _rodar(sorteios, simulacoes=2000):
    con = db.conectar(":memory:")
    db.gravar(con, sorteios)
    try:
        return backtest.backtest(con, sorteios[0].jogo, simulacoes=simulacoes,
                                 salvar_json=False, log=lambda *a, **k: None)
    finally:
        con.close()


class TestBacktest(unittest.TestCase):
    def test_uniforme_fica_dentro_do_ic(self):
        # histórico determinístico (seed fixa): sem vício, nenhuma estratégia
        # deve cair nos extremos da distribuição aleatória
        rng = np.random.default_rng(77)
        historia = [Sorteio(jogo="megasena", concurso=i + 1,
                            dezenas=sorted(int(d) + 1 for d in
                                           rng.choice(60, 6, replace=False)),
                            fonte="caixa")
                    for i in range(1200)]
        for r in _rodar(historia):
            self.assertTrue(0.5 < r.percentil < 99.5,
                            msg=f"{r.estrategia}: percentil {r.percentil}")

    def test_vicio_persistente_e_capturado(self):
        rng = np.random.default_rng(3)
        pesos = np.ones(60)
        pesos[:6] = 3.0  # seis dezenas com o triplo de peso, o tempo todo
        sorteios = []
        for i in range(2000):
            escolha = rng.choice(60, 6, replace=False, p=pesos / pesos.sum())
            sorteios.append(Sorteio(jogo="megasena", concurso=i + 1,
                                    dezenas=sorted(int(d) + 1 for d in escolha),
                                    fonte="caixa"))
        resultados = _rodar(sorteios)
        quentes = next(r for r in resultados if r.estrategia == "dezenas_quentes")
        self.assertGreater(quentes.percentil, 99.0)
        self.assertIn("FORA", quentes.veredito)

    def test_posicional(self):
        cfg = JOGOS["supersete"]
        historia = gerar_historia_sintetica(cfg, 600)
        resultados = _rodar(historia)
        self.assertEqual({r.estrategia for r in resultados},
                         {"digitos_quentes", "digitos_frios"})
        for r in resultados:
            self.assertEqual(len(r.aposta), 7)


if __name__ == "__main__":
    unittest.main()
