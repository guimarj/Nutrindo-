"""Volantes dos palpites em PDF, prontos para conferir na hora de marcar.

Lê relatorios/palpites_<jogo>.json (gerado pelo comando `palpites`) e desenha
um volante por jogo da carteira: grade completa do jogo com as dezenas do
palpite destacadas, mais scores, disputa e o rodapé de integridade.
"""

import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from .config import Jogo, jogo as _jogo

PASTA_RELATORIOS = Path("relatorios")

RODAPE = ("Otimizado para valor do premio, nao para chance de acerto. "
          "A probabilidade e a mesma de qualquer outro jogo.")


def _grade(cfg: Jogo) -> tuple[int, int]:
    """(colunas, linhas) do volante."""
    if cfg.posicional:
        return 10, cfg.dezenas_sorteadas       # 1 linha por coluna do Super Sete
    colunas = 5 if cfg.tamanho_universo == 25 else 10
    return colunas, (cfg.tamanho_universo + colunas - 1) // colunas


def _desenhar_volante(c: canvas.Canvas, x0: float, y0: float, cfg: Jogo,
                      dezenas: list[int], celula: float = 7.2 * mm):
    cols, linhas = _grade(cfg)
    marcadas = (set(enumerate(dezenas)) if cfg.posicional else set(dezenas))
    for linha in range(linhas):
        for col in range(cols):
            if cfg.posicional:
                valor, marcada = col, (linha, col) in marcadas
            else:
                valor = linha * cols + col + cfg.universo_min
                if valor > cfg.universo_max:
                    continue
                marcada = valor in marcadas
            x = x0 + col * celula
            y = y0 - linha * celula
            if marcada:
                c.setFillColorRGB(0.06, 0.20, 0.38)
                c.rect(x, y - celula, celula - 1, celula - 1, fill=1, stroke=0)
                c.setFillColorRGB(1, 1, 1)
            else:
                c.setFillColorRGB(1, 1, 1)
                c.setStrokeColorRGB(0.75, 0.75, 0.82)
                c.rect(x, y - celula, celula - 1, celula - 1, fill=0, stroke=1)
                c.setFillColorRGB(0.25, 0.25, 0.3)
            c.setFont("Helvetica-Bold" if marcada else "Helvetica", 8)
            c.drawCentredString(x + (celula - 1) / 2, y - celula + 2.2 * mm,
                                f"{valor:02d}")
    c.setFillColorRGB(0, 0, 0)


def gerar_pdf(slug: str, log=print) -> Path:
    origem = PASTA_RELATORIOS / f"palpites_{slug}.json"
    if not origem.exists():
        raise SystemExit(f"{origem} não existe; rode `palpites --jogo {slug}` antes")
    carteira = json.loads(origem.read_text(encoding="utf-8"))
    cfg = _jogo(slug)

    destino = PASTA_RELATORIOS / f"volantes_{slug}.pdf"
    c = canvas.Canvas(str(destino), pagesize=A4)
    largura, altura = A4
    cols, linhas_grade = _grade(cfg)
    celula = 7.2 * mm
    bloco_alt = linhas_grade * celula + 22 * mm
    y = altura - 30 * mm

    c.setFont("Helvetica-Bold", 16)
    c.drawString(20 * mm, altura - 18 * mm,
                 f"{cfg.nome} — carteira ({carteira['modo']}), "
                 f"custo R${carteira['custo_total']:.2f}")

    for i, p in enumerate(carteira["palpites"], 1):
        if y - bloco_alt < 25 * mm:
            _rodape_pagina(c, largura)
            c.showPage()
            y = altura - 20 * mm
        dezenas_txt = " ".join(f"{d:02d}" for d in p["dezenas"])
        c.setFont("Helvetica-Bold", 11)
        c.drawString(20 * mm, y, f"Jogo #{i:02d}:  {dezenas_txt}")
        c.setFont("Helvetica", 8)
        scores = ", ".join(f"{k}={v}" for k, v in p["scores"].items())
        c.drawString(20 * mm, y - 4.5 * mm,
                     f"{scores} | final={p['score_final']} | disputa "
                     f"{p['multiplicador_disputa']}x | prob. {p['probabilidade']}")
        _desenhar_volante(c, 20 * mm, y - 8 * mm, cfg, p["dezenas"], celula)
        y -= bloco_alt

    _rodape_pagina(c, largura)
    c.save()
    log(f"volantes salvos em {destino}")
    return destino


def _rodape_pagina(c: canvas.Canvas, largura: float):
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColorRGB(0.35, 0.35, 0.4)
    c.drawCentredString(largura / 2, 12 * mm, RODAPE)
    c.setFillColorRGB(0, 0, 0)
