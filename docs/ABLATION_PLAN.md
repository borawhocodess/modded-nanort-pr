# Plano de ablation — v3 (avaliado contra o código atual)

Revisão do plano v2 contra `simplicial_sigreg_new` @ `f29b46e`. Três achados no
código mudam a ordem das fases; um deles muda o objeto do estudo.

---

## Parte I — O que o código diz antes de qualquer run

### 1. A métrica reportada não é a métrica definida (bloqueante)

`README.md:71` define o alvo:

> Metric: **mean AUROC across `rel-avito` binary-classification tasks**, averaged over the val seeds.

`ValAUROCCallback` ([main.py:111-137](../rt/main.py#L111-L137)) acumula `preds`/`labels`
de **todos** os batches de **todas** as val tasks numa lista só e chama
`roc_auc_score` **uma vez**. Isso é AUROC **pooled**, não a média por tarefa.

Quais tarefas entram: `resolve_tasks` ([nanort_train.py:73-77](../scripts/nanort_train.py#L73-L77))
filtra por `f"{table}:Val" in table_info`. Em `~/scratch/pre/rel-avito/table_info.json`:

| task | chave | seeds de val |
|---|---|---|
| `user-visits` / `num_click` | `user-visits:Val` | 29 979 |
| `user-clicks` / `num_click` | `user-clicks:Val` | 21 183 |
| `SearchInfo` / `IsUserLoggedOn` | só `SearchInfo:Db` → **excluída** | — |

Então val = **2 tarefas com base rates diferentes**, num único AUROC pooled. E as
duas são distinguíveis na entrada da célula-alvo: `column_index.json` dá
`num_click of user-clicks` = 9 e `num_click of user-visits` = 3246415 — **índices de
texto distintos**, logo embeddings de `col_name` distintos, e `col_name` entra em `x`
incondicionalmente, inclusive na célula mascarada
([model.py:333-336](../rt/model.py#L333-L336)).

Consequência aritmética. Um score **constante dentro de cada tarefa**, ordenado pela
tarefa de maior base rate, carrega **zero** informação intra-tarefa e ainda assim
pontua, no pooled (n = 29 979 / 21 183):

```
AUROC_between = [n⁺_A·n⁻_B + ½(n⁺_A·n⁻_A + n⁺_B·n⁻_B)] / [(n⁺_A+n⁺_B)(n⁻_A+n⁻_B)]
```

| p(user-visits) ↓ / p(user-clicks) → | 0.05 | 0.10 | 0.20 | 0.30 |
|---|---|---|---|---|
| 0.60 | 0.786 | 0.754 | 0.697 | 0.646 |
| 0.70 | **0.822** | 0.794 | 0.743 | 0.695 |
| 0.80 | **0.864** | **0.840** | 0.794 | 0.751 |
| 0.90 | **0.916** | **0.896** | **0.857** | **0.821** |

**O alvo 0.81621 e o 0.87 reportado caem os dois dentro da faixa alcançável por pura
separação de base rate entre as duas tarefas.** Não é prova de que é isso que
acontece — é prova de que a métrica atual não consegue distinguir os dois casos.

Nota: o caminho de código que computa AUROC **por tarefa** já existe
([main.py:654-665](../rt/main.py#L654-L665)), mas `eval_tasks=[]`
([nanort_train.py:61](../scripts/nanort_train.py#L61)) e ele nunca roda.

Nota 2: se a calibração que produziu 0.81621 usou o mesmo pooled, os records 2 e 3
seguem **comparáveis entre si** e o speedrun continua internamente válido. O que cai
é a *interpretação* ("transferência relacional zero-shot"), não necessariamente o
ranking. Verificar `records/01_baseline` antes de afirmar qualquer coisa.

### 2. O reenquadramento "aprendido vs. init" já está escrito no código

[main.py:96-99](../rt/main.py#L96-L99):

> Skip the step-1 eval when False: on the static-relation route the near-init auroc
> is high (~0.87) and would trigger `time_to_target` at step 1, which is not a
> trained-model crossing.

Ou seja: **AUROC ≈ 0.87 no step 1 já é fato conhecido e documentado**, e o flag
`eval_at_step1` existe para impedir que `time_to_target` dispare nele. A hipótese
central do plano v2 (§"o confound primário é aprendido vs. no init") não precisa ser
testada — está confirmada no repositório.

Pior: o default do YAML é `eval_at_step1: "1"`
([plx_nanort_train.yaml:171](../plx_nanort_train.yaml#L171)). E `timer.start()` roda
**antes** do callback ([main.py:849-859](../rt/main.py#L849-L859)), então com o default
ligado `time_to_target` ≈ duração de um eval. Auditar qual valor o run do record usou.

Consequência: `time_to_target` neste branch mede **latência de infraestrutura**
(compile das flex masks — o próprio README diz ~3 min não-cronometrados — + download +
primeiro eval) e o *gap de 100 steps* do `val_eval_freq`. Não mede velocidade de
aprendizado. Isso precisa ser redefinido antes de ir pro paper.

Quanto os pesos andam até o step 25: Muon com warmup 3000 → LR(25) = 25/3000·5e-3 =
4.2e-5. E `enc_dict`/`dec_dict`/`mask_embs`/normas/LayerScale caem **todos** no grupo
AdamW ([optim/__init__.py](../rt/optim/__init__.py)), com OneCycle partindo de
`max_lr/25` = 4e-5. Nenhum dos dois moveu nada. Confirmado.

### 3. `relation_embs` não é parâmetro — a Fase de geometria é barata

[data.py:164-171](../rt/data.py#L164-L171): `relation_embs` é um tensor montado por DB
em `RelationalDataset.__init__` e entregue no dict do batch. Não é `nn.Parameter`;
"congelado" é por construção. Logo **todos** os arms `r-*` são uma transformação de
~20 linhas em `data.py`, aplicada por chunk de DB — o que automaticamente vale para
treino e para o holdout, porque são duas instâncias do mesmo `Dataset`. Sem tocar
modelo, sem tocar compile.

Detalhe: `R_train = Σ_DBs` mas `R_val = R_holdout` (offsets em
[data.py:105-113](../rt/data.py#L105-L113)), então `num_relations` **muda** entre treino
e eval. Já logado como `simplicial/r_table_size`.

Detalhe 2: `r_mask = torch.ones(...)` ([data.py:171](../rt/data.py#L171)) — o re-mask
defensivo de [model.py:357-358](../rt/model.py#L357-L358) é no-op hoje.

---

## Parte II — Correções técnicas ao plano v2

| item v2 | correção |
|---|---|
| 0.A.2 "o termo `k̄` é absorvível em `W_q`/`W_k`" | **Mais forte e exato**: `q_norm` é `nn.RMSNorm(d_head)` com `elementwise_affine=True`, logo `q̂ = w ⊙ (W_q x / rms)`. `diag(k̄)` funde-se **exatamente** em `q_norm.weight`. A componente-média do operador é reparametrização pura. |
| "…portanto redundante com o `feat`" | **Não** é redundante com `feat`: o suporte da softmax é outro (`feat` normaliza sobre a linha densa `same_node \| kv_in_f2p`; `nbr_attn` sobre o conjunto esparso de arestas FK por pai). A afirmação correta é "reduz a atenção bilinear sobre as arestas FK" — que é o **controle** certo, não uma redundância. |
| 0.A.3 SVD de `E ∈ R^{R×d_head}` | Fazer **por head**: `kr` é `(R,H,D)` = `(R,8,32)` pós-`kr_norm`. `rank(E_h) ≤ min(R,32)`. Com `R_avito` ≈ 20 (contagem de FK do schema), **não é low-rank por construção** — o argumento "R pequeno ⇒ rank baixo" só morde se `R ≲ 5`. Continua valendo como medida empírica. |
| 0.A.4 "`qk_norm` destrói a geometria global" | Provavelmente **falso no init**, e a previsão deve estar escrita antes: `W_kr` é 256→256 densa, então cada head vê uma projeção aleatória de 256→32 do vetor de relação; por JL a projeção **preserva cosseno** aproximadamente, e a RMSNorm remove só o raio. Previsão: correlação alta (≳0.8). Se der baixa, é achado; se der alta, o arm `r-gram` pode casar a Gram pré-norm. |
| ausente | **O gargalo de contraste está antes do `wkr`**: `enc_dict.relation` é `Linear(384,256, bias=True)`. Com texto unit-norm, ‖Wx‖ ≈ 0.47 e ‖b‖ ≈ 0.47 no init — **bias e sinal da mesma ordem**. Depois vem `norm_dict.relation` (RMSNorm global, mata o raio). Ou seja `r⁰ ≈ (viés comum) + (parte específica de relação de mesma magnitude)`, e as frases-FK (`"{col} between {child} and {parent}"`) já são mutuamente muito similares em MiniLM. **Previsão: `r-const ≈ r-text` e a Fase de geometria dá nulo.** Isso é mensurável a custo zero e é o primeiro gate. |
| D3 (`R` passes exatos de flex_attention) | **Morto por aritmética.** Custo `B·R·S²` = 256·20·1024² ≈ 5.4e9 vs `T` ≈ 1e5–4e5 arestas (`valid_fk_slots ≤ B·S·5` = 1.3M, com `parent_match_rate` < 1). Quatro ordens de grandeza pior. Remover do plano. |
| D2 marcado "✓ fusível" | **Não é fusível em `score_mod`.** `score_mod` recebe um escalar de score e índices — dá para somar um termo por `(relação, head)` (isso é D1), mas não para computar `q̂·(k̂ ⊙ e_r)`, que precisa dos vetores. D1 ✓, D2 ✗. |
| D1 | **Já existe**: branch `relation-score-mod` (bias de relação via `score_mod`). Reaproveitar em vez de reimplementar. A rota de fusão real é `flex_attention` com a máscara `kv_in_f2p` (já construída em [model.py:292](../rt/model.py#L292)) + `score_mod` indexando um `(R,H)` — o id de relação por par `(q,kv)` cabe em `int8` se `R<127`: 256·1024²·1B = 268 MB. |
| 0.C "`r-const` é o 'off' correto" | Concordo, e vale reforçar: desligar `simplicial_layers=none` remove módulos (muda contagem de params e grupos do Muon), `r-const` não. Manter `r-const` como o off canônico. |
| variância | Falta uma fonte, e ela é grande: **a rede toda é bf16** ([main.py:427](../rt/main.py#L427)), então os scores de val têm ~3 dígitos decimais e há muitos **empates**; `roc_auc_score` os resolve por média. Medir a fração de empates. Somar a isso: `index_add_` em [simplicial_attention.py:135,189](../rt/simplicial_attention.py#L135) é atômico ⇒ não-determinístico na GPU. Já o **subgrafo de val está pinado** (`seed=0` + `shuffle_py(0)`, `__getitem__` puro em `batch_idx`, `in_order=True`) — essa fonte pode ser descartada de antemão. |
| orçamento | Boa parte da Fase 0 é *código*, não GPU. `R` por DB já sai no log de todo run (`[artifact-check] … relation_types=`, [yaml:265-271](../plx_nanort_train.yaml#L265)), e `active_relation_types` já vai pro wandb. Nada de novo a rodar para 0.A.1. |

---

## Parte III — Plano revisado

Métrica primária **redefinida** para toda a série: `mean_per_task_AUROC` — a métrica
do README. `pooled_AUROC` é reportado ao lado, rotulado como *métrica legada do
speedrun*, para manter comparabilidade com os records 2 e 3.

### Fase 0 — Decomposição da métrica e do init (nenhum treino)

Gate para todo o resto. Custo: quase todo em código.

**0.1 — Decompor o AUROC pooled. ✅ IMPLEMENTADO** ([rt/val_metrics.py](../rt/val_metrics.py),
ligado em [main.py:161](../rt/main.py#L161); testes em
[tests/test_val_metrics.py](../tests/test_val_metrics.py), rodam sem torch/GPU).

O id de tarefa já estava no batch: `col_name_idxs[is_targets]` (9 = `num_click of
user-clicks`, 3246415 = `num_click of user-visits`), mapeado por `get_column_index`. Zero
plumbing novo. Logado por eval:

- `val/auroc_zero_shot` — pooled, **intocado**: é contra ele que os records e o
  `time_to_target` são medidos
- `val/auroc/<task>` e `val/auroc_mean_per_task` ← **primária** (a métrica do README)
- `val/base_rate/<task>`, `val/n_pos/<task>`, `val/n_neg/<task>`
- `val/auroc_between` — fórmula fechada acima, teto de base-rate puro
- `val/auroc_pooled_within`, `val/auroc_pooled_cross`, `val/pooled_cross_weight` —
  split **exato** do pooled: como AUROC é uma média de Mann-Whitney sobre pares
  (positivo, negativo), ela fatora sobre a tarefa de origem de cada elemento, e vale a
  identidade `pooled = (1−cw)·within + cw·cross` (verificada no teste a 1e-9).
  `cross_weight` é a fração da métrica reportada que repousa em pares **entre** tarefas —
  comparações que o benchmark nunca pede. Nas contagens de val do avito, `cw ≈ 0.72`.
- `val/pred_tie_frac`, `val/pred_unique` — piso de ruído do bf16
- `val/n_tasks`, `val/n_unmapped_tasks` — se o mapa de id falhar, o split continua
  correto (agrupa pelo id observado), só sem nome resolvido

Descartado no caminho: a primeira versão logava "pooled após centrar os scores por
tarefa". Centrar pela média do grupo **não** remove o efeito de base rate — a média do
grupo depende dela, e o resultado ficou 0.70 contra 0.86 de `mean_per_task` num fixture
onde o sinal intra-tarefa era idêntico nos dois grupos. Sem interpretação; substituído
pelo split exato de Mann-Whitney.

Custo GPU: um eval. **Gate G0**: se `auroc_mean_per_task ≪ auroc_zero_shot ≈
auroc_between`, o "0.87 zero-shot" é separação de base rate e as Fases 2–4 mudam de
objeto — passam a medir efeito sobre `auroc_mean_per_task`, que pode estar em ~0.5.

Como obter a leitura sem gastar 45 min: `eval_at_step1=1` (default do YAML) + `max_steps`
pequeno já entrega a decomposição no step 1, que é ~init.

**0.2 — `AUROC@0` e sua distribuição sobre seeds.** Uma linha para habilitar
(`val_callback.step(0, net)` antes do loop de treino; `0 % eval_every == 0` já passa o
guard de [main.py:107](../rt/main.py#L107)), mais um flag para impedir
`time_to_target` de disparar no step 0. Forward-only, ~1 min por seed, **20 seeds**:

- distribuição de `AUROC@0` (pooled e per-task), com **sinal**: fração de seeds com AUROC < 0.5
- variante *head-only*: fixar a seed do backbone e variar só `dec_dict`/`mask_embs`

Este é o experimento que separa duas histórias muito diferentes. Se a distribuição está
centrada em ~0.87 com spread pequeno, a representação de entrada separa o rótulo quase
independentemente do readout → o achado é sobre dados+encoder, não sobre o ramo de
relação (e é um baseline *training-free*, potencialmente publicável). Se está centrada
em 0.5 com cauda larga e simétrica, o `best-by-val-AUROC` sobre uma curva decrescente
está apenas escolhendo o máximo do ruído — história oposta, e mais grave.

**0.3 — Reprodutibilidade.** Config champion, mesma seed, 3×. Comparar `AUROC@0` e a
curva. Fontes a atribuir: bf16/empates (0.1), `index_add_` atômico, e — descartada de
antemão — amostragem de val. Se o spread for da ordem de 0.01, nenhuma comparação de
arm com Δ < 0.02 é afirmável. Auditar em paralelo a config real do run 0.87 contra o
`.md` (`eval_at_step1`, `sigreg_layer`, `muon_warmup_steps`).

**0.4 — Geometria da relação no init.** Job de 1 min (precisa dos artefatos do cluster;
os locais em `~/scratch/pre/` são do pré-processamento antigo, sem
`relation_text_idxs.json`). Medir, nesta ordem, porque cada etapa pode zerar o
contraste:

1. cossenos par-a-par de `relation_embs` (texto puro)
2. após `enc_dict.relation` + `norm_dict.relation` → razão ‖b‖/‖Wx‖ e cossenos
3. após `wkr` + `kr_norm`, por head → cossenos, e correlação com (1)
4. `k̄` vs `e_r`: `‖e_r‖/‖k̄‖`
5. SVD **por head** de `E_h`, espectro e rank efetivo

**Gate G1**: se o contraste em (2) já for desprezível, `r-const ≈ r-text` é previsão, não
achado, e a Fase 3 só vale para confirmar o nulo com 1–2 arms.

**0.5 — `R` e cobertura.** Puro wandb/log: `relation_types` por DB do `[artifact-check]`,
`simplicial/r_table_size`, `active_relation_types`, `active_relation_type_frac`,
`parent_match_rate`, `edge_count`. Zero compute.

### Fase 1 — Protocolo (pré-registro)

- **Primária**: `mean_per_task_AUROC @ K`, com `K` fixo pré-registrado. Escolher `K` a
  partir da curva do champion, **não** por arm.
- Reportar sempre o quarteto: `@0` · `@K` · `@peak` (rotulado *oráculo*) · `t2t`
  (rotulado *inclui startup*), mais `pooled` legado.
- **Pareamento**: mesma seed, mesma ordem de dados, mesmo stream de init entre arms;
  reportar diferenças pareadas. Com 3 seeds isso é a diferença entre ter e não ter poder.
- **Variância**: eval (SE analítico + empates do bf16), seed (spread), kernel
  (`index_add_`). O efeito mínimo detectável sai daí e vai no pré-registro.
- **`val_eval_freq`**: 100 é grosso demais para um pico no step ~25. Usar 5 nos arms curtos.
- **s/step por variante**: 50 steps, sem eval, sem checkpoint, mais `edges/s`.

### Fase 2 — Qual canal de texto carrega o sinal

Gate: G0. Grid em step-0 (**barato: ~1 min/célula, 10+ seeds**) e depois em treino curto.

`{r-text, r-perm, r-const} × {c-text, c-perm, c-const}`

- `r-*`: transformação em `data.py` sobre cada chunk de DB (ver Parte I §3).
- `c-*`: `col_name_idxs` são linhas da tabela de texto e `column_index.json` diz quais
  linhas são nomes de coluna → permutação/constante por DB, re-gather de
  `col_name_values` como transformação de batch em `main.py`.
- Permutação = bijeção fixa **dentro** de cada DB, mesmo procedimento em treino e holdout.

**Atenção**: `c-perm` sobre os nomes de coluna **destrói a distinguibilidade entre as duas
val tasks** se as duas linhas de `num_click` forem trocadas entre si — o que colapsa
`auroc_between`. Isso é feature, não bug: é exatamente o teste do confound do §1. Mas
tem de ser reportado com `auroc_between` ao lado, senão o Δ é ilegível.

Leitura: contribuição de cada canal + interação. Se `AUROC(r-const, c-text) ≈
AUROC(r-text, c-text)`, o ramo de relação não contribui no init.

### Fase 3 — Geometria ou identidade

Gate: `r-const < r-text` na Fase 2 (e G1 não ter previsto o nulo). Se o gate falhar,
rodar só `r-const` + `r-orth` para documentar o nulo e parar.

| arm | `relation_embs` | isola |
|---|---|---|
| `r-const` | vetor único compartilhado | piso |
| `r-text` | texto (champion) | referência |
| `r-perm` | bijeção fixa por DB | similaridade ↔ papel relacional |
| `r-orth` | ortogonalizado, pares equidistantes | toda a estrutura de similaridade, mantendo identidade |
| `r-gram` | aleatório com Gram casada **pós-`kr_norm`** | similaridade estatística sem semântica |
| `r-anon` | `"rel_7 between t3 and t9"` | léxico vs. formato de schema |
| `r-struct` | direção, cardinalidade, grau do pai, dtype → projeção linear | alternativa transferível, sem texto |

Restrições que a arquitetura impõe: `kr_norm` per-head remove o raio, então **arms que
diferem só em norma são no-op por construção**. `r-gram`/`r-orth` têm de gerar vetores
também para os tipos de FK do holdout, senão o teste é memorização e não transferência.
`r-anon` e `r-struct` exigem embedar texto novo → `rt/embed.py` (SentenceTransformer,
barato em CPU) num passo de pré-processamento. `r-struct` exige metadados de FK que
hoje não estão em `table_info.json` — **custo maior, agendar por último**.

Matriz de decisão: manter a do v2, com a ressalva de que todas as linhas passam a ser
lidas em `mean_per_task_AUROC`, não em pooled.

### Fase 4 — Decomposição do operador

Gate: `r-const ≪ r-text` **em `mean_per_task_AUROC`**.

Com `k̂^rel_r = k̄ + e_r`:

```
ℓ = (q̂ ⊙ k̄)·k̂_s /√d    ← funde exatamente em q_norm.weight → atenção bilinear sobre arestas FK
  + (q̂ ⊙ e_r)·k̂_s /√d   ← todo o sinal específico de relação
```

| arm | operador | fusível |
|---|---|---|
| `D0` | trilinear + scatter (champion) | — |
| `D1` | rank-1 `e_r ≈ c_r·u` → escalar por (relação, head) | ✓ `score_mod` — **já em `relation-score-mod`** |
| `D2` | rank-J via SVD por head (J de 0.4) | ✗ score_mod; só no scatter esparso |
| `D4` | gate só no value, score sem `r` | ✓ |
| `D5` | key aditiva `q̂·(k̂ + k̂_r)` | ✓ |
| `D6` | adaLN: pool de `r` nas arestas → scale+shift no bloco | ✓ |
| `D7` | cross-attn cells→relations (`S×R`) | ✓ |
| `D8` | champion com `simplicial_layerscale_init=0` | — |
| `Dbil` | **novo**: só o termo `k̄` (controle bilinear correto) | ✓ |

`D3` do v2 removido (aritmética, Parte II). `Dbil` é o controle que faltava: separa
"o operador trilinear ajuda" de "atenção sobre arestas FK ajuda".

`D8` testa o reenquadramento direto: com LayerScale init 0 o modelo no step 0 **é**
`r-const`, e o ramo tem de conquistar entrada. Se nunca recupera 0.87, confirma artefato
de init. Nota de implementação: `layerscale_init=0` mantém gradiente para os params do
`nbr_attn`? `dx = attn_out * nbr_scale` com `nbr_scale=0` → `∂L/∂W = 0` no primeiro step;
só `∂L/∂nbr_scale ≠ 0`. Sai do zero, mas devagar. Usar `1e-2` em vez de `0` como arm
adicional para não confundir "não ajuda" com "não treina" (é o mesmo modo de falha que o
commit `430b5d4` já corrigiu no `RelAttn`).

### Fase 5 — Custo e redefinição do `t2t`

- Fronteira `mean_per_task_AUROC @ K` × `s/step`.
- AUROC sob budget fixo de 45 min de wall-clock, fiel à competição.
- **Redefinir `t2t`**: reportar `t2t_total` (atual) decomposto em `startup`
  (download + build rustler + compile das flex masks, ~3 min pelo README) +
  `train` + `eval_overhead` (os evals contam no `TrainTimer`). E reportar
  `AUROC@0` ao lado, para expor quanto do `t2t` é a rede nem ter treinado.
- Se `AUROC@0 ≥ 0.81621` em `pooled`, dizer explicitamente que o `t2t` legado é
  degenerado neste regime.

### Fase 6 — Condicional, não é ablation

Só se a Fase 3 der positivo: relation dropout estilo CFG; testar se SIGReg **para** o
decay (agora mensurável como distância ao init, com `AUROC@0` na mão); softmax com
segmentação conjunta (endpoint livre) como direção de pesquisa fora do caminho crítico.

---

## Orçamento

| fase | o que é | GPU-h |
|---|---|---|
| 0.1 | código (~40 linhas) + 1 eval | ~0.05 |
| 0.2 | 20 seeds forward-only, step 0 | ~0.4 |
| 0.3 | 3 runs curtos (~10 min) | ~0.5 |
| 0.4 | 1 job de análise | ~0.05 |
| 0.5 | wandb/logs | 0 |
| 1 | reaproveita 0.2/0.3 + ~6 curtos | ~1.0 |
| 2 | 9 células × 10 seeds step-0, + 9 × 3 curtos | ~3.5 |
| 3 | ≤7 arms × 3 seeds × 10 min (gated) | ~3.5 |
| 4 | ≤8 arms × 3 seeds × 45 min (gated) | ~18 |
| 5 | reaproveita | ~1 |

**Caminho crítico: 0.1 → 0.4 → 0.2.** Decompor a métrica, medir o contraste da relação
no init, medir a distribuição de `AUROC@0`. É uma tarde, e as Fases 2–4 só têm objeto se
esses três não encerrarem a discussão.

---

## Disciplina estatística

Pré-registrar antes de rodar: `K`, lista de arms, métrica primária
(`mean_per_task_AUROC`), efeito mínimo detectável vindo de 0.3, e os gates G0/G1. Com
~20 arms e picos de oráculo, vencedores espúrios são garantidos. Tudo fora do
pré-registro é exploratório e vai rotulado como tal.

## Os papers possíveis, reordenados

1. **A métrica de zero-shot no RelBench multi-task é dominada por base rate** — se G0
   confirmar, isto é um resultado metodológico sobre como esse tipo de avaliação é
   reportado, e vale independente de tudo o mais aqui. É o mais provável e o mais barato.
2. **Random features + texto de schema dão zero-shot forte sem treino** — se `AUROC@0`
   for alto e estável em `mean_per_task_AUROC` (não só em pooled). Baseline
   *training-free*, e reenquadra `t2t` como métrica de infraestrutura.
3. **Conditioning por tipo de aresta transfere zero-shot** (`r-orth ≈ r-text`) — claim
   mais fraco que o original mas limpo; `r-struct` dispensa o encoder de texto.
4. **Rank baixo recupera o trilinear a custo bilinear** (`D2` com `J` pequeno ≈ `D0`,
   contra `Dbil`) — contribuição de método.
5. **Semântica lexical genuína** (`r-perm` e `r-anon` caem, em `mean_per_task_AUROC`) —
   o claim original. Menos provável, dado o §1 e o gate G1.
