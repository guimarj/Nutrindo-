"""Definições estruturais de cada loteria da Caixa.

`posicional=True` (Super Sete) significa que a ordem das dezenas identifica a
coluna do volante e dígitos podem se repetir entre colunas; nos demais jogos as
dezenas são um subconjunto sem reposição do universo.

Os preços de aposta mudam por portaria da Caixa — confira o valor vigente
antes de usar o modo --orcamento.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Jogo:
    slug: str                 # identificador usado na API da Caixa e no espelho
    nome: str
    universo_min: int
    universo_max: int
    dezenas_sorteadas: int
    aposta_minima: int        # dezenas na aposta simples
    aposta_maxima: int        # máximo de dezenas por volante
    preco_aposta_simples: float
    posicional: bool = False
    faixas_premiadas: tuple = field(default_factory=tuple)  # acertos que pagam prêmio

    @property
    def universo(self) -> range:
        return range(self.universo_min, self.universo_max + 1)

    @property
    def tamanho_universo(self) -> int:
        return self.universo_max - self.universo_min + 1


JOGOS: dict[str, Jogo] = {
    "megasena": Jogo(
        slug="megasena", nome="Mega-Sena",
        universo_min=1, universo_max=60, dezenas_sorteadas=6,
        aposta_minima=6, aposta_maxima=20, preco_aposta_simples=6.00,
        faixas_premiadas=(6, 5, 4),
    ),
    "lotofacil": Jogo(
        slug="lotofacil", nome="Lotofácil",
        universo_min=1, universo_max=25, dezenas_sorteadas=15,
        aposta_minima=15, aposta_maxima=20, preco_aposta_simples=3.50,
        faixas_premiadas=(15, 14, 13, 12, 11),
    ),
    "quina": Jogo(
        slug="quina", nome="Quina",
        universo_min=1, universo_max=80, dezenas_sorteadas=5,
        aposta_minima=5, aposta_maxima=15, preco_aposta_simples=2.50,
        faixas_premiadas=(5, 4, 3, 2),
    ),
    "supersete": Jogo(
        slug="supersete", nome="Super Sete",
        universo_min=0, universo_max=9, dezenas_sorteadas=7,
        aposta_minima=7, aposta_maxima=21, preco_aposta_simples=2.50,
        posicional=True,
        faixas_premiadas=(7, 6, 5, 4, 3),
    ),
}


def jogo(slug: str) -> Jogo:
    try:
        return JOGOS[slug]
    except KeyError:
        raise ValueError(
            f"Jogo desconhecido: {slug!r}. Opções: {', '.join(JOGOS)}"
        ) from None
