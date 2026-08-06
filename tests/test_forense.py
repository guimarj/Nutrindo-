"""Testes da bateria forense.

Dois eixos: (1) CALIBRAÇÃO — sob H0 (dados sintéticos uniformes) a bateria
não deve rejeitar; (2) SENSIBILIDADE — vícios plantados devem ser detectados.
Seeds fixas para não haver flakiness.
"""

import math
import unittest
from itertools import combinations
from math import comb

import numpy as np

from loteria import forense
from loteria.config import JOGOS
from loteria.db import Sorteio
from loteria.forense import bitstream, nist
from loteria.forense.resultado import bh_qvalores


def _sorteios_uniformes(cfg, m, seed):
    rng = np.random.default_rng(seed)
    if cfg.posicional:
        listas = rng.integers(0, 10, size=(m, 7)).tolist()
    else:
        listas = [sorted(rng.choice(cfg.tamanho_universo,
                                    cfg.dezenas_sorteadas,
                                    replace=False) + cfg.universo_min)
                  for _ in range(m)]
    return [Sorteio(jogo=cfg.slug, concurso=i + 1,
                    dezenas=[int(d) for d in dz], fonte="caixa")
            for i, dz in enumerate(listas)]


class TestBH(unittest.TestCase):
    def test_bh_exemplo_conhecido(self):
        # exemplo clássico: q_(i) = min_{j>=i} p_(j)*n/j
        p = np.array([0.01, 0.04, 0.03, 0.005])
        q = bh_qvalores(p)
        np.testing.assert_allclose(q, [0.02, 0.04, 0.04, 0.02])

    def test_bh_monotono_e_limitado(self):
        rng = np.random.default_rng(1)
        p = rng.random(100)
        q = bh_qvalores(p)
        self.assertTrue((q >= p - 1e-12).all())
        self.assertTrue((q <= 1.0).all())


class TestBitstream(unittest.TestCase):
    def test_posto_cobre_intervalo_sem_colisao(self):
        # todas as C(6,3)=20 combinações de um universo pequeno
        postos = {bitstream.posto_combinacao(list(c))
                  for c in combinations(range(1, 7), 3)}
        self.assertEqual(postos, set(range(comb(6, 3))))

    def test_derivacao_uniforme(self):
        rng = np.random.default_rng(7)
        listas = [sorted(rng.choice(60, 6, replace=False) + 1)
                  for _ in range(3000)]
        bits, taxa = bitstream.derivar_bits(listas, 60, 6)
        # aceitação teórica 2^25/C(60,6) ≈ 0.67
        self.assertAlmostEqual(taxa, 0.67, delta=0.05)
        # monobit não deve rejeitar bits derivados de dados uniformes
        r = nist.bateria_nist(bits)
        monobit = next(x for x in r if "monobit" in x.teste)
        self.assertGreater(monobit.p_valor, 0.001)


class TestNist(unittest.TestCase):
    def test_padrao_constante_reprova(self):
        bits = np.ones(10000, dtype=np.uint8)
        monobit = next(r for r in nist.bateria_nist(bits)
                       if "monobit" in r.teste)
        self.assertLess(monobit.p_valor, 1e-10)

    def test_alternancia_reprova_runs_mas_nao_monobit(self):
        bits = np.tile([0, 1], 5000).astype(np.uint8)
        rs = nist.bateria_nist(bits)
        monobit = next(r for r in rs if "monobit" in r.teste)
        runs = next(r for r in rs if r.teste == "runs")
        self.assertGreater(monobit.p_valor, 0.9)
        self.assertLess(runs.p_valor, 1e-10)

    def test_bits_aleatorios_passam(self):
        rng = np.random.default_rng(42)
        bits = rng.integers(0, 2, 50000, dtype=np.uint8)
        for r in nist.bateria_nist(bits):
            self.assertGreater(r.p_valor, 1e-4,
                               msg=f"{r.teste} rejeitou bits uniformes")


class TestCalibracao(unittest.TestCase):
    """Sob H0, nenhuma família deve rejeitar (com seed fixa)."""

    def _checar(self, slug, m, seed):
        cfg = JOGOS[slug]
        sorteios = _sorteios_uniformes(cfg, m, seed)
        resultados = forense.rodar_bateria(sorteios, cfg,
                                           incluir_trincas=(slug != "quina"))
        resumos = forense.resumir(resultados, alfa=0.01)
        rejeicoes = [r for r in resumos if "REJEITA" in r.veredito]
        self.assertEqual(rejeicoes, [],
                         msg=f"{slug}: {[(r.familia, r.q_minimo) for r in rejeicoes]}")

    def test_megasena_sintetica(self):
        self._checar("megasena", 2000, seed=101)

    def test_supersete_sintetico(self):
        self._checar("supersete", 600, seed=103)


class TestSensibilidade(unittest.TestCase):
    def test_dezena_viciada_e_detectada(self):
        """Dezena 07 com o dobro da probabilidade: chi2 global deve rejeitar."""
        cfg = JOGOS["megasena"]
        rng = np.random.default_rng(55)
        pesos = np.ones(60)
        pesos[6] = 2.0
        listas = []
        for _ in range(2000):
            p = pesos / pesos.sum()
            escolha = rng.choice(60, 6, replace=False, p=p)
            listas.append(sorted(int(d) + 1 for d in escolha))
        sorteios = [Sorteio(jogo="megasena", concurso=i + 1, dezenas=dz,
                            fonte="caixa") for i, dz in enumerate(listas)]
        resultados = forense.rodar_bateria(sorteios, cfg, incluir_trincas=False)
        chi2 = next(r for r in resultados
                    if r.familia == "chi2_frequencia")
        self.assertLess(chi2.q_valor, 0.001)

    def test_periodicidade_e_detectada(self):
        """Dezena que sai a cada 7 concursos: espectral deve acusar."""
        cfg = JOGOS["megasena"]
        rng = np.random.default_rng(56)
        listas = []
        for t in range(2000):
            if t % 7 == 0:
                resto = rng.choice(59, 5, replace=False) + 2  # 2..60
                listas.append(sorted([1] + [int(d) for d in resto]))
            else:
                escolha = rng.choice(59, 6, replace=False) + 2
                listas.append(sorted(int(d) for d in escolha))
        sorteios = [Sorteio(jogo="megasena", concurso=i + 1, dezenas=dz,
                            fonte="caixa") for i, dz in enumerate(listas)]
        resultados = forense.rodar_bateria(sorteios, cfg, incluir_trincas=False)
        espec = [r for r in resultados if r.familia == "espectral"
                 and "dezena 01" in r.teste]
        self.assertTrue(espec and espec[0].q_valor < 0.001)


if __name__ == "__main__":
    unittest.main()
