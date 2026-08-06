"""Fonte oficial: API pública do portal de loterias da Caixa.

Endpoint: https://servicebus2.caixa.gov.br/portaldeloterias/api/{jogo}[/{concurso}]

É a única fonte com metadados completos (data, acumulado, ganhadores por
faixa, arrecadação), mas exige uma requisição por concurso — a carga inicial
completa é lenta e deve ser feita uma única vez; depois a atualização é
incremental. Alguns ambientes corporativos/cloud bloqueiam esse host; nesse
caso use a fonte espelho e enriqueça os metadados quando possível.
"""

import time

import requests

from ..db import Sorteio
from .base import FonteDados

_BASE = "https://servicebus2.caixa.gov.br/portaldeloterias/api"
_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) loteria-suite/0.1",
}


class FonteCaixa(FonteDados):
    nome = "caixa"

    def __init__(self, pausa_s: float = 0.20, tentativas: int = 3,
                 timeout_s: float = 20.0):
        self.pausa_s = pausa_s
        self.tentativas = tentativas
        self.timeout_s = timeout_s
        self.sessao = requests.Session()
        self.sessao.headers.update(_HEADERS)

    def disponivel(self) -> bool:
        try:
            r = self.sessao.get(f"{_BASE}/megasena", timeout=self.timeout_s)
            return r.ok
        except requests.RequestException:
            return False

    def buscar(self, jogo_slug: str, apos_concurso: int = 0,
               ate_concurso: int | None = None) -> list[Sorteio]:
        mais_recente = self._obter(jogo_slug)
        if mais_recente is None:
            return []
        ultimo = ate_concurso or int(mais_recente["numero"])
        sorteios = []
        for n in range(apos_concurso + 1, ultimo + 1):
            bruto = (mais_recente if n == int(mais_recente["numero"])
                     else self._obter(jogo_slug, n))
            if bruto is None:
                continue  # a lacuna fica registrada; `lacunas()` a expõe
            sorteios.append(self._converter(jogo_slug, bruto))
            time.sleep(self.pausa_s)
        return sorteios

    def buscar_um(self, jogo_slug: str, concurso: int) -> Sorteio | None:
        bruto = self._obter(jogo_slug, concurso)
        return None if bruto is None else self._converter(jogo_slug, bruto)

    def _obter(self, jogo_slug: str, concurso: int | None = None) -> dict | None:
        url = f"{_BASE}/{jogo_slug}" + (f"/{concurso}" if concurso else "")
        for tentativa in range(self.tentativas):
            try:
                r = self.sessao.get(url, timeout=self.timeout_s)
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 404:
                    return None
            except (requests.RequestException, ValueError):
                pass
            time.sleep(2 ** tentativa)
        return None

    @staticmethod
    def _converter(jogo_slug: str, bruto: dict) -> Sorteio:
        # ordem de sorteio quando disponível; Lotofácil por exemplo não a expõe
        dezenas = bruto.get("dezenasSorteadasOrdemSorteio") or bruto["listaDezenas"]
        premios = [
            {
                "faixa": f.get("faixa"),
                "descricao": f.get("descricaoFaixa"),
                "ganhadores": f.get("numeroDeGanhadores"),
                "valor": f.get("valorPremio"),
            }
            for f in bruto.get("listaRateioPremio") or []
        ]
        return Sorteio(
            jogo=jogo_slug,
            concurso=int(bruto["numero"]),
            dezenas=[int(d) for d in dezenas],
            data=_data_iso(bruto.get("dataApuracao")),
            acumulado=bruto.get("acumulado"),
            arrecadacao=bruto.get("valorArrecadado"),
            premios=premios or None,
            fonte="caixa",
        )


def _data_iso(data_br: str | None) -> str | None:
    if not data_br:
        return None
    try:
        d, m, a = data_br.split("/")
        return f"{a}-{m}-{d}"
    except ValueError:
        return None
