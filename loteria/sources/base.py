from abc import ABC, abstractmethod

from ..db import Sorteio


class FonteDados(ABC):
    """Uma origem de resultados. Implementações devem ser incrementais:
    `buscar` recebe o último concurso já persistido e retorna só o que falta."""

    nome: str

    @abstractmethod
    def disponivel(self) -> bool:
        """Checagem barata de alcançabilidade (uma requisição no máximo)."""

    @abstractmethod
    def buscar(self, jogo_slug: str, apos_concurso: int = 0,
               ate_concurso: int | None = None) -> list[Sorteio]:
        """Resultados com concurso > apos_concurso (e <= ate_concurso, se dado)."""
