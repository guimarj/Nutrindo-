import math
import unittest

import numpy as np

from loteria import popularidade
from loteria.config import JOGOS
from loteria.controle import gerar_historia_sintetica
from loteria.popularidade import irls_poisson, montar_modelo


class TestIRLS(unittest.TestCase):
    def test_recupera_betas_conhecidos(self):
        rng = np.random.default_rng(11)
        n, p = 3000, 3
        X = rng.normal(size=(n, p))
        offset = np.log(rng.uniform(50, 150, n))
        beta_verdadeiro = np.array([0.5, -0.3, 0.15])
        mu = np.exp(offset + X @ beta_verdadeiro)
        y = rng.poisson(mu).astype(float)
        beta = irls_poisson(X, y, offset)
        np.testing.assert_allclose(beta, beta_verdadeiro, atol=0.05)


class TestModelo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = JOGOS["megasena"]
        cls.historia = gerar_historia_sintetica(cls.cfg, 400)
        cls.modelo = montar_modelo(cls.historia, cls.cfg)

    def test_fallback_sem_metadados(self):
        self.assertIn("fallback", self.modelo.origem)

    def test_normalizacao_media_proxima_de_um(self):
        rng = np.random.default_rng(5)
        mults = []
        for _ in range(3000):
            dez = sorted((rng.choice(60, 6, replace=False) + 1).tolist())
            mults.append(self.modelo.multiplicador(dez, self.cfg))
        self.assertAlmostEqual(float(np.mean(mults)), 1.0, delta=0.15)

    def test_combinacao_popular_tem_multiplicador_alto(self):
        seq_datas = self.modelo.multiplicador([1, 2, 3, 4, 5, 6], self.cfg)
        datas = self.modelo.multiplicador([3, 7, 11, 19, 23, 30], self.cfg)
        espalhada = self.modelo.multiplicador([4, 17, 33, 41, 48, 57], self.cfg)
        self.assertGreater(seq_datas, 5 * espalhada)
        self.assertGreater(datas, espalhada)

    def test_supersete_digito_repetido_e_popular(self):
        cfg = JOGOS["supersete"]
        modelo = montar_modelo(gerar_historia_sintetica(cfg, 100), cfg)
        tudo_sete = modelo.multiplicador([7] * 7, cfg)
        variado = modelo.multiplicador([3, 8, 1, 6, 0, 9, 4], cfg)
        self.assertGreater(tudo_sete, 2 * variado)


if __name__ == "__main__":
    unittest.main()
