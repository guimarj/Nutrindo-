"""Relatório HTML autossuficiente (CSS e SVG inline, sem dependências
externas): p-valores e veredito por família, dashboard real × sintético e
resultado do backtest, por jogo. Lê os JSONs gerados pelos comandos
`analisar`, `controle` e `backtest`."""

import html
import json
from pathlib import Path

from .config import JOGOS

PASTA_RELATORIOS = Path("relatorios")

_CSS = """
body{font-family:system-ui,sans-serif;margin:2rem auto;max-width:1100px;
     color:#1a1a2e;background:#fafafa;line-height:1.45}
h1{border-bottom:3px solid #0f3460}h2{color:#0f3460;margin-top:2.5rem}
table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.9rem}
th,td{border:1px solid #d0d0e0;padding:.35rem .6rem;text-align:left}
th{background:#0f3460;color:#fff}tr:nth-child(even){background:#f0f0f8}
.ok{color:#1a7a4a;font-weight:600}.rej{color:#b02a2a;font-weight:600}
.falso{color:#a06a00;font-weight:600}
.aviso{background:#fff6e0;border-left:4px solid #d99000;padding:.7rem 1rem;
       margin:1rem 0}
.rodape{margin-top:3rem;padding-top:1rem;border-top:1px solid #ccc;
        font-size:.85rem;color:#555}
svg{background:#fff;border:1px solid #d0d0e0;margin:.5rem 0}
"""


def _carregar(nome: str) -> dict | None:
    caminho = PASTA_RELATORIOS / nome
    if not caminho.exists():
        return None
    return json.loads(caminho.read_text(encoding="utf-8"))


def _fmt_p(p: float) -> str:
    return f"{p:.2e}" if p < 0.001 else f"{p:.4f}"


def _barras_svg(itens: list[tuple[str, float, float]], titulo: str) -> str:
    """Barras horizontais pareadas: q real vs mediana sintética (escala
    -log10, truncada em 4)."""
    def altura(q):
        return min(4.0, -__import__("math").log10(max(q, 1e-4)))

    lg, alt_linha = 260, 22
    w = 720
    h = len(itens) * alt_linha + 50
    partes = [f'<svg width="{w}" height="{h}" role="img" '
              f'aria-label="{html.escape(titulo)}">']
    partes.append(f'<text x="10" y="18" font-weight="600">{html.escape(titulo)}'
                  f' — barras: −log₁₀(q); vermelho=real, cinza=sintético (mediana)</text>')
    escala = (w - lg - 60) / 4.0
    for i, (nome, q_real, q_sint) in enumerate(itens):
        y = 40 + i * alt_linha
        partes.append(f'<text x="10" y="{y + 10}" font-size="12">'
                      f'{html.escape(nome)}</text>')
        partes.append(f'<rect x="{lg}" y="{y}" width="{altura(q_real) * escala:.1f}" '
                      f'height="8" fill="#b02a2a"/>')
        partes.append(f'<rect x="{lg}" y="{y + 9}" width="{altura(q_sint) * escala:.1f}" '
                      f'height="8" fill="#9a9ab0"/>')
    # linha de significância q=0.05
    x_sig = lg + altura(0.05) * escala
    partes.append(f'<line x1="{x_sig:.1f}" y1="34" x2="{x_sig:.1f}" y2="{h - 6}" '
                  f'stroke="#d99000" stroke-dasharray="4 3"/>')
    partes.append(f'<text x="{x_sig + 4:.1f}" y="{h - 8}" font-size="11" '
                  f'fill="#d99000">q=0,05</text>')
    partes.append("</svg>")
    return "".join(partes)


def gerar_html(log=print) -> Path:
    secoes = []
    for slug, cfg in JOGOS.items():
        forense = _carregar(f"forense_{slug}.json")
        controle = _carregar(f"controle_{slug}.json")
        backtest = _carregar(f"backtest_{slug}.json")
        if not any([forense, controle, backtest]):
            continue
        s = [f"<h2>{html.escape(cfg.nome)}</h2>"]

        if forense:
            s.append(f"<p>{forense['concursos']} concursos analisados; "
                     f"FDR (Benjamini-Hochberg) por família, "
                     f"α={forense['alfa']}.</p>")
            s.append("<table><tr><th>família</th><th>testes</th>"
                     "<th>min p</th><th>min q</th><th>veredito</th>"
                     "<th>destaque</th></tr>")
            for r in sorted(forense["resumos"], key=lambda r: r["q_minimo"]):
                classe = "rej" if "REJEITA" in r["veredito"] else "ok"
                s.append(f"<tr><td>{html.escape(r['familia'])}</td>"
                         f"<td>{r['n_testes']}</td>"
                         f"<td>{_fmt_p(r['p_minimo'])}</td>"
                         f"<td>{_fmt_p(r['q_minimo'])}</td>"
                         f"<td class='{classe}'>{html.escape(r['veredito'])}</td>"
                         f"<td>{html.escape(r['destaque'])}</td></tr>")
            s.append("</table>")

        if controle:
            s.append(f"<h3>Controle negativo ({controle['replicas']} réplicas "
                     "de RNG criptográfico)</h3>")
            itens = [(c["familia"], c["q_min_real"],
                      c["q_min_sintetico_mediana"])
                     for c in sorted(controle["comparacoes"],
                                     key=lambda c: c["q_min_real"])]
            s.append(_barras_svg(itens, f"{cfg.nome}: real × sintético"))
            s.append("<table><tr><th>família</th><th>q real</th>"
                     "<th>q sintético (mediana)</th><th>rejeições sintéticas</th>"
                     "<th>p empírico</th><th>veredito</th></tr>")
            for c in sorted(controle["comparacoes"],
                            key=lambda c: c["q_min_real"]):
                if c["veredito"].startswith("ACHADO"):
                    classe = "rej"
                elif "falso" in c["veredito"]:
                    classe = "falso"
                else:
                    classe = "ok"
                p_emp = "—" if c["p_empirico"] is None else f"{c['p_empirico']:.3f}"
                s.append(f"<tr><td>{html.escape(c['familia'])}</td>"
                         f"<td>{_fmt_p(c['q_min_real'])}</td>"
                         f"<td>{_fmt_p(c['q_min_sintetico_mediana'])}</td>"
                         f"<td>{c['taxa_rejeicao_sintetica']:.0%}</td>"
                         f"<td>{p_emp}</td>"
                         f"<td class='{classe}'>{html.escape(c['veredito'])}"
                         f"</td></tr>")
            s.append("</table>")

        if backtest:
            s.append(f"<h3>Backtest (treino 1-{backtest['n_treino']}, "
                     f"teste em {backtest['n_teste']} concursos seguintes, "
                     f"{backtest['simulacoes']} apostas aleatórias)</h3>")
            s.append("<table><tr><th>estratégia</th><th>média de acertos</th>"
                     "<th>média aleatória</th><th>IC95% aleatório</th>"
                     "<th>percentil</th><th>veredito</th></tr>")
            for r in backtest["resultados"]:
                classe = "rej" if "FORA" in r["veredito"] else "ok"
                ic = r["ic95_media"]
                s.append(f"<tr><td>{html.escape(r['estrategia'])}</td>"
                         f"<td>{r['media_acertos']:.4f}</td>"
                         f"<td>{r['media_acertos_aleatoria']:.4f}</td>"
                         f"<td>[{ic[0]:.4f}, {ic[1]:.4f}]</td>"
                         f"<td>{r['percentil']:.1f}%</td>"
                         f"<td class='{classe}'>{html.escape(r['veredito'])}"
                         f"</td></tr>")
            s.append("</table>")
        secoes.append("".join(s))

    corpo = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<title>Loterias Caixa — análise forense de aleatoriedade</title>
<style>{_CSS}</style></head><body>
<h1>Loterias Caixa — análise forense de aleatoriedade</h1>
<div class="aviso"><strong>Regras de integridade.</strong> Nenhum p-valor é
reportado sem correção para múltiplas comparações (FDR de
Benjamini-Hochberg por família). Um desvio só é tratado como achado se
rejeitar H0 nos dados reais E não rejeitar nas réplicas sintéticas de RNG
criptográfico. Resultados in-sample nunca são apresentados como preditivos —
o backtest com split temporal estrito é quem decide, e resultado negativo é
reportado com o mesmo destaque do positivo.</div>
{"".join(secoes)}
<div class="rodape">Gerado por <code>python loteria.py relatorio</code>.
Qualquer palpite derivado desta análise é otimizado para valor do prêmio,
não para chance de acerto — a probabilidade é a mesma de qualquer outro
jogo.</div>
</body></html>"""

    PASTA_RELATORIOS.mkdir(exist_ok=True)
    destino = PASTA_RELATORIOS / "relatorio.html"
    destino.write_text(corpo, encoding="utf-8")
    log(f"relatório salvo em {destino}")
    return destino
