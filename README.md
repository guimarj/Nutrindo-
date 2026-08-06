# Suíte de análise das loterias da Caixa

Análise estatística forense dos sorteios da Mega-Sena, Lotofácil, Quina e
Super Sete, terminando num gerador de jogos otimizado para **valor
condicional do prêmio** — nunca para chance de acerto, que é matematicamente
idêntica para qualquer aposta.

## Instalação e uso rápido

```bash
pip install -r requirements.txt

python loteria.py atualizar                  # baixa/atualiza o histórico
python loteria.py status                     # resumo do banco
python loteria.py analisar  --jogo megasena  # Fase 2: bateria forense
python loteria.py controle  --jogo megasena  # Fase 3: controle negativo
python loteria.py backtest  --jogo megasena  # Fase 4: split temporal
python loteria.py disputa   --jogo megasena --dezenas 01,02,03,04,05,06
python loteria.py palpites  --jogo megasena --n 10
python loteria.py relatorio                  # HTML com tudo
python loteria.py volantes  --jogo megasena  # PDF dos volantes
```

Modos do gerador: `--modo conservador | contrarian | fechamento`,
`--orcamento 50`, `--universo 12` (fechamento), `--seed` (reprodutibilidade).

## Dados

Duas fontes plugáveis, com atualização incremental em SQLite
(`dados/loterias.sqlite`):

| fonte | conteúdo | quando usar |
|---|---|---|
| API oficial da Caixa (`servicebus2.caixa.gov.br`) | dezenas + data, acumulado, **ganhadores por faixa**, arrecadação | rede aberta; necessária para ajustar o modelo de popularidade aos dados |
| espelho GitHub ([guilhermeasn/loteria.json](https://github.com/guilhermeasn/loteria.json)) | só dezenas (ordem de sorteio), atualização diária | carga inicial rápida; redes que bloqueiam o host da Caixa |

Um registro da API oficial sempre pode enriquecer um registro do espelho;
o inverso nunca sobrescreve. `atualizar --fonte caixa` completa os metadados
de um banco populado pelo espelho.

## Metodologia

**Fase 2 — bateria forense.** Nove grupos de testes com p-valor:
qui-quadrado por dezena (global e janelas móveis) e teste G de entropia, com
o ajuste de Joe (1993) para amostragem sem reposição; KS de uniformidade
(jitter contínuo, seed fixa); Ljung-Box e runs de Wald-Wolfowitz por série
indicadora; periodograma com teste g de Fisher; coocorrência de pares e
trincas contra a binomial exata; gaps contra a geométrica; subconjunto da
bateria NIST SP 800-22 (8 testes) sobre bitstream derivado por posto
combinatório com rejeição (bits exatamente uniformes sob H0); teste de
compressibilidade por Monte Carlo. **Toda família recebe correção FDR de
Benjamini-Hochberg antes de qualquer veredito** — na coocorrência, sobre o
vetor completo de C(N,2)/C(N,3) combinações.

**Fase 3 — controle negativo.** Réplicas do histórico geradas com
`secrets.SystemRandom` (RNG criptográfico) passam pela mesma bateria. Um
desvio real só vira achado se `p_emp = (1+#{q_sint ≤ q_real})/(R+1) ≤ α`
— confirmar a α=0,05 exige R≥19 réplicas.

**Fase 4 — backtest.** Estratégias treinadas apenas no passado (dezenas
quentes, frias, trinca quente) avaliadas nos concursos seguintes contra
10.000 apostas aleatórias no mesmo período (IC 95%). Resultado observado nos
quatro jogos: **nenhuma estratégia sai do intervalo do acaso** — os desvios
estatísticos detectados não são preditivos.

**Fase 5 — popularidade.** Modelo log-linear da densidade de apostas sobre
padrões que a população joga (datas ≤31, sequências, progressões, soma
baixa, concentração no volante, dezenas quentes, repetição do último
resultado). Com metadados de ganhadores no banco, os pesos são ajustados por
regressão de Poisson (IRLS, offset de volume); sem eles, usa pesos
documentados da literatura (Simon 1998; Farrell & Walker 1999; Riedwyl
2002). O multiplicador é normalizado para média 1 sobre combinações
uniformes.

**Fase 6 — gerador.** Score multicritério transparente: anti-rateio (peso
maior), aderência ao perfil histórico (soma no IQR, paridade, faixas),
diversificação da carteira (seleção gulosa com penalidade de sobreposição) e
— somente quando as Fases 2-3 confirmam desvio de frequência — um componente
de desvio com peso proporcional ao tamanho de efeito (teto 0,15). O modo
`fechamento` monta cobertura gulosa com **verificação exaustiva** da
garantia de quadra.

## Resultados da análise (agosto/2026)

- **Super Sete**: indistinguível de RNG criptográfico em todas as famílias.
- **Quina**: desvio confirmado apenas numa janela histórica (concursos
  2251–2500) — assinatura de troca de equipamento/processo no passado, sem
  valor preditivo; excluído do gerador por construção.
- **Mega-Sena / Lotofácil**: desvio pequeno mas estatisticamente confirmado
  na frequência global (w=0,069 e 0,022). O backtest mostra que **não é
  explorável**: estratégia de dezenas quentes fica dentro do IC do acaso.

## Limitações — leia antes de usar

1. **Nenhum jogo aqui aumenta a chance de ganhar.** A probabilidade de uma
   aposta simples é a mesma de qualquer outra (Mega-Sena: 1 em 50.063.860).
   O gerador otimiza apenas quanto você levaria *se* ganhasse, evitando
   combinações populares que rateiam o prêmio.
2. O modelo de popularidade em modo fallback usa pesos heurísticos da
   literatura internacional; a intensidade real dos padrões no Brasil só é
   estimável com os metadados de ganhadores da API oficial.
3. Prêmio condicional e volume de apostas são estimativas grosseiras e
   configuráveis (`--premio`, `--volume`); preços de aposta mudam por
   portaria da Caixa — confira `loteria/config.py`.
4. O espelho GitHub contém apenas dezenas; erros de digitação na fonte
   produziriam exatamente o tipo de desvio de frequência detectado. Antes de
   interpretar achados como física dos sorteios, enriqueça o banco com a
   fonte oficial e re-rode `analisar` + `controle`.
5. Desvios confirmados em janelas históricas antigas não dizem nada sobre o
   próximo concurso.
6. Loteria é entretenimento com valor esperado negativo. Jogue apenas o que
   pode perder.

## Regras de integridade (aplicadas no código)

- Nenhum padrão reportado sem correção para múltiplas comparações.
- Nenhum resultado in-sample apresentado como preditivo; o backtest reporta
  resultado negativo com o mesmo destaque do positivo.
- Todo palpite exibe a probabilidade real e o rodapé:
  *"Otimizado para valor do prêmio, não para chance de acerto. A
  probabilidade é a mesma de qualquer outro jogo."*

## Testes

```bash
python -m unittest discover -s tests
```

Cobrem: persistência e precedência de fontes; calibração da bateria sob H0 e
sensibilidade a vícios plantados; sanidade NIST; veredito do controle
negativo nos dois sentidos; calibração e poder do backtest; recuperação de
parâmetros do IRLS; garantia do fechamento (verificação exaustiva);
validade, diversificação e orçamento das carteiras.
