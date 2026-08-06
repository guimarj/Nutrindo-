import unittest

from loteria import db
from loteria.db import Sorteio


def _mega(concurso, dezenas, **kw):
    return Sorteio(jogo="megasena", concurso=concurso, dezenas=dezenas, **kw)


class TestGravacao(unittest.TestCase):
    def setUp(self):
        self.con = db.conectar(":memory:")

    def tearDown(self):
        self.con.close()

    def test_insercao_e_incremental(self):
        res = db.gravar(self.con, [
            _mega(1, [4, 5, 30, 33, 41, 52], fonte="github_mirror"),
            _mega(2, [9, 37, 39, 41, 43, 49], fonte="github_mirror"),
        ])
        self.assertEqual(res.inseridos, 2)
        self.assertEqual(db.ultimo_concurso(self.con, "megasena"), 2)

    def test_espelho_nao_sobrescreve_caixa(self):
        db.gravar(self.con, [_mega(1, [1, 2, 3, 4, 5, 6], fonte="caixa",
                                   data="2020-01-01", acumulado=True)])
        res = db.gravar(self.con, [_mega(1, [1, 2, 3, 4, 5, 6],
                                         fonte="github_mirror")])
        self.assertEqual(res.ignorados, 1)
        s = db.carregar(self.con, "megasena")[0]
        self.assertEqual(s.fonte, "caixa")
        self.assertEqual(s.data, "2020-01-01")

    def test_caixa_enriquece_espelho(self):
        db.gravar(self.con, [_mega(1, [1, 2, 3, 4, 5, 6], fonte="github_mirror")])
        res = db.gravar(self.con, [_mega(
            1, [1, 2, 3, 4, 5, 6], fonte="caixa", data="2020-01-01",
            premios=[{"faixa": 1, "ganhadores": 0, "valor": 0.0}])])
        self.assertEqual(res.enriquecidos, 1)
        s = db.carregar(self.con, "megasena")[0]
        self.assertEqual(s.fonte, "caixa")
        self.assertEqual(s.premios[0]["faixa"], 1)

    def test_validacao_rejeita_dezenas_erradas(self):
        res = db.gravar(self.con, [
            _mega(1, [1, 2, 3, 4, 5]),            # faltou dezena
            _mega(2, [1, 2, 3, 4, 5, 61]),        # fora do universo
            _mega(3, [1, 1, 3, 4, 5, 6]),         # repetida
        ])
        self.assertEqual(res.inseridos, 0)
        self.assertEqual(len(res.invalidos), 3)

    def test_supersete_permite_digitos_repetidos(self):
        s = Sorteio(jogo="supersete", concurso=1,
                    dezenas=[6, 2, 1, 9, 3, 9, 2], fonte="github_mirror")
        res = db.gravar(self.con, [s])
        self.assertEqual(res.inseridos, 1)

    def test_lacunas(self):
        db.gravar(self.con, [_mega(1, [1, 2, 3, 4, 5, 6], fonte="github_mirror"),
                             _mega(4, [7, 8, 9, 10, 11, 12], fonte="github_mirror")])
        self.assertEqual(db.lacunas(self.con, "megasena"), [2, 3])


if __name__ == "__main__":
    unittest.main()
