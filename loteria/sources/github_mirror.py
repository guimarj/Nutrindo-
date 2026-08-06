"""Fonte espelho: repositório público guilhermeasn/loteria.json.

Um arquivo JSON por jogo com TODAS as dezenas históricas (ordem de sorteio
preservada), atualizado diariamente por GitHub Actions. Uma única requisição
baixa o histórico completo — ideal para carga inicial e para ambientes onde o
host da Caixa é bloqueado.

Limitação: só dezenas. Data, acumulado, ganhadores e arrecadação ficam NULL e
podem ser enriquecidos depois pela fonte oficial (ver precedência em db.py).
"""

import requests

from ..db import Sorteio
from .base import FonteDados

_BASE = "https://raw.githubusercontent.com/guilhermeasn/loteria.json/master/data"


class FonteEspelhoGithub(FonteDados):
    nome = "github_mirror"

    def __init__(self, timeout_s: float = 60.0):
        self.timeout_s = timeout_s
        self.sessao = requests.Session()

    def disponivel(self) -> bool:
        try:
            r = self.sessao.head(f"{_BASE}/megasena.json", timeout=self.timeout_s)
            return r.ok
        except requests.RequestException:
            return False

    def buscar(self, jogo_slug: str, apos_concurso: int = 0,
               ate_concurso: int | None = None) -> list[Sorteio]:
        r = self.sessao.get(f"{_BASE}/{jogo_slug}.json", timeout=self.timeout_s)
        r.raise_for_status()
        dados: dict[str, list] = r.json()
        sorteios = []
        for chave, dezenas in dados.items():
            n = int(chave)
            if n <= apos_concurso or (ate_concurso and n > ate_concurso):
                continue
            sorteios.append(Sorteio(
                jogo=jogo_slug, concurso=n,
                dezenas=[int(d) for d in dezenas],
                fonte=self.nome,
            ))
        sorteios.sort(key=lambda s: s.concurso)
        return sorteios
