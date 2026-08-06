"""Testes do controle negativo: sintético vs sintético não confirma nada;
vício plantado sobrevive ao controle."""

import unittest

import numpy as np

from loteria import controle, db, forense
from loteria.config import JOGOS
from loteria.controle import (VEREDITO_CONFIRMADO, gerar_historia_sintetica)
from loteria.db import Sorteio


class TestGeracao(unittest.TestCase):
    def test_historia_sintetica_valida(self):
        cfg = JOGOS["megasena"]
        h = gerar_historia_sintetica(cfg, 50)
        self.assertEqual(len(h), 50)
        for s in h:
            self.assertEqual(len(s.dezenas), 6)
            self.assertEqual(len(set(s.dezenas)), 6)
            self.assertTrue(all(1 <= d <= 60 for d in s.dezenas))

    def test_historia_supersete_valida(self):
        cfg = JOGOS["supersete"]
        h = gerar_historia_sintetica(cfg, 30)
        for s in h:
            self.assertEqual(len(s.dezenas), 7)
            self.assertTrue(all(0 <= d <= 9 for d in s.dezenas))


class TestVeredito(unittest.TestCase):
    def _rodar_controle(self, sorteios, replicas=6):
        """Roda o controle contra um banco em memória com esses sorteios."""
        con = db.conectar(":memory:")
        db.gravar(con, sorteios)
        try:
            return controle.controle_negativo(
                con, sorteios[0].jogo, replicas=replicas,
                incluir_trincas=False, salvar_json=False,
                log=lambda *a, **k: None)
        finally:
            con.close()

    def test_sintetico_contra_sintetico_nao_confirma(self):
        cfg = JOGOS["megasena"]
        reais = gerar_historia_sintetica(cfg, 700)
        comparacoes = self._rodar_controle(reais)
        confirmados = [c for c in comparacoes
                       if c.veredito == VEREDITO_CONFIRMADO]
        self.assertEqual(confirmados, [],
                         msg=[(c.familia, c.q_min_real) for c in confirmados])

    def test_vicio_plantado_sobrevive_ao_controle(self):
        rng = np.random.default_rng(9)
        pesos = np.ones(60)
        pesos[6] = 2.2  # dezena 07 fortemente viciada
        sorteios = []
        for i in range(1500):
            escolha = rng.choice(60, 6, replace=False, p=pesos / pesos.sum())
            sorteios.append(Sorteio(jogo="megasena", concurso=i + 1,
                                    dezenas=sorted(int(d) + 1 for d in escolha),
                                    fonte="caixa"))
        # 20 réplicas: o mínimo para que p_emp = 1/(R+1) fique <= 0.05
        comparacoes = self._rodar_controle(sorteios, replicas=20)
        confirmadas = {c.familia for c in comparacoes
                       if c.veredito == VEREDITO_CONFIRMADO}
        self.assertIn("chi2_frequencia", confirmadas)


if __name__ == "__main__":
    unittest.main()
