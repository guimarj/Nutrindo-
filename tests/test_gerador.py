import unittest

from loteria import db, gerador
from loteria.config import JOGOS
from loteria.controle import gerar_historia_sintetica
from loteria.gerador import (RODAPE, fechamento_quadra, gerar_carteira,
                             verificar_fechamento)


def _con_com_historia(slug, m=400):
    con = db.conectar(":memory:")
    db.gravar(con, gerar_historia_sintetica(JOGOS[slug], m))
    return con


class TestFechamento(unittest.TestCase):
    def test_garantia_de_quadra(self):
        universo = [4, 17, 23, 35, 41, 47, 50, 52, 56, 59]
        jogos = fechamento_quadra(universo, 6)
        self.assertTrue(verificar_fechamento(universo, jogos))
        for j in jogos:
            self.assertEqual(len(j), 6)
            self.assertTrue(set(j) <= set(universo))

    def test_verificador_detecta_cobertura_incompleta(self):
        universo = list(range(1, 11))
        # um único jogo não cobre todos os quintetos de 10 dezenas
        self.assertFalse(verificar_fechamento(universo, [universo[:6]]))


class TestCarteira(unittest.TestCase):
    def test_orcamento_define_n(self):
        con = _con_com_historia("megasena")
        try:
            carteira = gerar_carteira(con, "megasena", n=99, orcamento=31.0,
                                      seed=1, pool=800)
        finally:
            con.close()
        # R$31 / R$6 = 5 apostas simples
        self.assertEqual(len(carteira.palpites), 5)
        self.assertEqual(carteira.custo_total, 30.0)
        self.assertEqual(carteira.rodape, RODAPE)

    def test_palpites_validos_e_diversos(self):
        con = _con_com_historia("megasena")
        try:
            carteira = gerar_carteira(con, "megasena", n=6, seed=2, pool=800)
        finally:
            con.close()
        vistos = []
        for p in carteira.palpites:
            self.assertEqual(len(set(p.dezenas)), 6)
            self.assertTrue(all(1 <= d <= 60 for d in p.dezenas))
            self.assertEqual(p.probabilidade, "1 em 50.063.860")
            vistos.append(set(p.dezenas))
        # nenhuma dupla de jogos pode ser idêntica
        for i in range(len(vistos)):
            for j in range(i + 1, len(vistos)):
                self.assertLess(len(vistos[i] & vistos[j]), 6)

    def test_contrarian_evita_disputa(self):
        con = _con_com_historia("megasena")
        try:
            contra = gerar_carteira(con, "megasena", n=5, modo="contrarian",
                                    seed=3, pool=800)
        finally:
            con.close()
        for p in contra.palpites:
            self.assertLess(p.multiplicador_disputa, 1.0)

    def test_modo_fechamento_integra(self):
        con = _con_com_historia("megasena")
        try:
            carteira = gerar_carteira(con, "megasena", modo="fechamento",
                                      universo_fechamento=10, seed=4)
        finally:
            con.close()
        universo = sorted({d for p in carteira.palpites for d in p.dezenas})
        jogos = [p.dezenas for p in carteira.palpites]
        self.assertTrue(verificar_fechamento(universo, jogos))

    def test_supersete(self):
        con = _con_com_historia("supersete", m=200)
        try:
            carteira = gerar_carteira(con, "supersete", n=4, seed=5, pool=500)
        finally:
            con.close()
        for p in carteira.palpites:
            self.assertEqual(len(p.dezenas), 7)
            self.assertTrue(all(0 <= d <= 9 for d in p.dezenas))
            self.assertEqual(p.probabilidade, "1 em 10.000.000")


if __name__ == "__main__":
    unittest.main()
