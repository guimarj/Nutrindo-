"""Persistência em SQLite com atualização incremental.

Regra de precedência entre fontes: a API oficial da Caixa carrega metadados
(data, acumulado, ganhadores por faixa, arrecadação) que o espelho GitHub não
tem, então um registro vindo de `caixa` sempre pode sobrescrever um registro
vindo de `github_mirror`; o inverso nunca acontece. Assim o banco pode ser
populado rápido pelo espelho e enriquecido depois, concurso a concurso, quando
a API oficial estiver alcançável.
"""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

DB_PADRAO = Path("dados/loterias.sqlite")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS concursos (
    jogo          TEXT    NOT NULL,
    concurso      INTEGER NOT NULL,
    data          TEXT,              -- ISO yyyy-mm-dd; NULL quando a fonte não informa
    dezenas       TEXT    NOT NULL,  -- JSON, ordem de sorteio preservada
    acumulado     INTEGER,           -- 1/0; NULL quando a fonte não informa
    arrecadacao   REAL,
    premios       TEXT,              -- JSON [{faixa, descricao, ganhadores, valor}]
    fonte         TEXT    NOT NULL,  -- 'caixa' | 'github_mirror'
    atualizado_em TEXT    NOT NULL,
    PRIMARY KEY (jogo, concurso)
);
CREATE INDEX IF NOT EXISTS idx_concursos_jogo_data ON concursos (jogo, data);
"""

_PRIORIDADE_FONTE = {"github_mirror": 0, "caixa": 1}


@dataclass
class Sorteio:
    jogo: str
    concurso: int
    dezenas: list[int]
    data: str | None = None
    acumulado: bool | None = None
    arrecadacao: float | None = None
    premios: list[dict] | None = None  # [{faixa, descricao, ganhadores, valor}]
    fonte: str = "caixa"


@dataclass
class ResultadoIngestao:
    inseridos: int = 0
    enriquecidos: int = 0   # espelho -> caixa
    ignorados: int = 0      # já existia com fonte igual ou melhor
    invalidos: list[str] = field(default_factory=list)


def conectar(caminho: Path | str = DB_PADRAO) -> sqlite3.Connection:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(caminho)
    con.row_factory = sqlite3.Row
    con.executescript(_SCHEMA)
    return con


def ultimo_concurso(con: sqlite3.Connection, jogo: str) -> int:
    row = con.execute(
        "SELECT MAX(concurso) AS m FROM concursos WHERE jogo = ?", (jogo,)
    ).fetchone()
    return row["m"] or 0


def concursos_sem_metadados(con: sqlite3.Connection, jogo: str) -> list[int]:
    """Concursos presentes só via espelho — candidatos a enriquecimento."""
    rows = con.execute(
        "SELECT concurso FROM concursos WHERE jogo = ? AND fonte != 'caixa' "
        "ORDER BY concurso", (jogo,)
    ).fetchall()
    return [r["concurso"] for r in rows]


def gravar(con: sqlite3.Connection, sorteios: list[Sorteio]) -> ResultadoIngestao:
    from .config import jogo as _jogo

    res = ResultadoIngestao()
    agora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for s in sorteios:
        erro = _validar(s, _jogo(s.jogo))
        if erro:
            res.invalidos.append(f"{s.jogo} #{s.concurso}: {erro}")
            continue
        atual = con.execute(
            "SELECT fonte FROM concursos WHERE jogo = ? AND concurso = ?",
            (s.jogo, s.concurso),
        ).fetchone()
        if atual is not None:
            if _PRIORIDADE_FONTE[s.fonte] <= _PRIORIDADE_FONTE[atual["fonte"]]:
                res.ignorados += 1
                continue
            res.enriquecidos += 1
        else:
            res.inseridos += 1
        con.execute(
            "INSERT OR REPLACE INTO concursos "
            "(jogo, concurso, data, dezenas, acumulado, arrecadacao, premios, fonte, atualizado_em) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                s.jogo, s.concurso, s.data, json.dumps(s.dezenas),
                None if s.acumulado is None else int(s.acumulado),
                s.arrecadacao,
                None if s.premios is None else json.dumps(s.premios, ensure_ascii=False),
                s.fonte, agora,
            ),
        )
    con.commit()
    return res


def carregar(con: sqlite3.Connection, jogo: str) -> list[Sorteio]:
    rows = con.execute(
        "SELECT * FROM concursos WHERE jogo = ? ORDER BY concurso", (jogo,)
    ).fetchall()
    return [
        Sorteio(
            jogo=r["jogo"], concurso=r["concurso"],
            dezenas=json.loads(r["dezenas"]), data=r["data"],
            acumulado=None if r["acumulado"] is None else bool(r["acumulado"]),
            arrecadacao=r["arrecadacao"],
            premios=None if r["premios"] is None else json.loads(r["premios"]),
            fonte=r["fonte"],
        )
        for r in rows
    ]


def lacunas(con: sqlite3.Connection, jogo: str) -> list[int]:
    """Concursos faltando na sequência 1..max — integridade da série."""
    rows = con.execute(
        "SELECT concurso FROM concursos WHERE jogo = ? ORDER BY concurso", (jogo,)
    ).fetchall()
    presentes = {r["concurso"] for r in rows}
    if not presentes:
        return []
    return [n for n in range(1, max(presentes) + 1) if n not in presentes]


def _validar(s: Sorteio, cfg) -> str | None:
    if len(s.dezenas) != cfg.dezenas_sorteadas:
        return f"esperava {cfg.dezenas_sorteadas} dezenas, veio {len(s.dezenas)}"
    fora = [d for d in s.dezenas if not (cfg.universo_min <= d <= cfg.universo_max)]
    if fora:
        return f"dezenas fora do universo: {fora}"
    if not cfg.posicional and len(set(s.dezenas)) != len(s.dezenas):
        return f"dezenas repetidas: {sorted(s.dezenas)}"
    if s.fonte not in _PRIORIDADE_FONTE:
        return f"fonte desconhecida: {s.fonte}"
    return None
