# SIGReg no nanoRT — documentação da implementação

Implementação de **SIGReg (Sketched Isotropic Gaussian Regularization)** no
nanoRT, adaptando o método de *"Optimizing Pre-Training of Tabular Foundation
Models by Shaping Geometry"* (que usa nanoTabPFN) e do LeJEPA
([Balestriero & LeCun, 2025, arXiv:2511.08544](https://arxiv.org/abs/2511.08544)).

A hipótese do paper: regularizar a **geometria das representações intermediárias
em uma camada inicial** para que fiquem mais isotrópicas/gaussianas acelera a
convergência, melhora a performance de classificação e a estabilidade entre
seeds — com ~10–15 % de overhead por passo, mas reduzindo os passos até
convergir quase pela metade. No contexto do *speedrun* nanoRT, o que interessa é
o **time-to-target** (`train/time_to_target`): SIGReg só vale a pena se a
convergência mais rápida pagar o overhead por passo.

---

## 1. O desafio: nanoRT é relacional, não um grid

No paper, o hidden state do nanoTabPFN é um grid limpo `H ∈ R^{B×R×C×E}`
(B tabelas, R linhas, C colunas, E embedding). O paper define duas *views*:

- **column view** — agrupa os R datapoints de uma mesma feature (coluna);
- **row view** — agrupa as C features de um mesmo datapoint (linha).

No nanoRT o hidden state é uma **sequência achatada de células**
`x ∈ (B, S, E)` (S = `seq_len`), sem grid linha×coluna. Recuperamos as duas
views da estrutura relacional que o modelo **já calcula**:

| View do paper | nanoRT — agrupamento | Campo do batch |
|---|---|---|
| **column view** (datapoints de uma feature) | células com mesma `(table_name_idx, col_name_idx)` | `table_name_idxs`, `col_name_idxs` — é exatamente o que a atenção `col` opera |
| **row view** (features de um datapoint) | células com mesmo `node_idx` | `node_idxs` — parte da atenção `feat` |

Cada view tem seu próprio *projection head* e a estatística SIGReg é avaliada
**por grupo** e agregada com peso proporcional ao número de células válidas.

---

## 2. A estatística SIGReg (sketched Epps–Pulley)

Portada fiel da implementação oficial do LeJEPA (`EppsPulley` + slicing). Para
uma população de embeddings `Z ∈ R^{N×D}`:

1. Amostra **K direções 1-D aleatórias** `A ∈ R^{D×K}` (gaussianas, normalizadas
   à esfera unitária).
2. Projeta: `P = Z @ A ∈ R^{N×K}`.
3. Para cada projeção, compara a **função característica empírica**
   `φ̂(t) = mean_n exp(i·t·P_n)` com a da normal padrão `φ_N(t) = exp(−t²/2)`,
   integrada contra a janela gaussiana `w(t) = exp(−t²/2)`:

   ```
   T = N · ∫ |φ̂(t) − φ_N(t)|² w(t) dt
     = N · Σ_t [ (mean_n cos(t·P_n) − e^{−t²/2})² + (mean_n sin(t·P_n))² ] · weights_t
   ```

   Integração trapezoidal em `t ∈ [0, t_max=3]` com 17 nós (a evenness do
   integrando dispensa o lado negativo). **Não há padronização separada** dos
   dados: o próprio teste empurra a distribuição para `N(0,1)` (média 0, var 1).

4. Por grupo: média sobre as K projeções. Entre grupos: **média ponderada por
   N_g** (a estatística de cada grupo é escalada por seu N_g e a soma é dividida
   pelo total de células válidas, não pelo número de grupos). Isso torna a perda
   **invariante à escala do agrupamento** — ver §3 sobre o pooling do col view.

Quanto **menor** o valor, mais gaussiana/isotrópica está a representação
projetada.

---

## 3. O que foi alterado

### Arquivo novo: `rt/sigreg.py`
- **`MaskedBatchNorm1d`** — BatchNorm que calcula média/variância **só sobre as
  células válidas** (ignora padding). Sem running stats: SIGReg só roda em
  treino, então sempre usa estatísticas do batch.
- **`Projector`** — MLP `E → 4·E → E/4` com `MaskedBatchNorm` + ReLU
  (estilo VICReg/LeJEPA; o paper indica que o projector é **crítico**). Um
  projector independente por view (`row_proj`, `col_proj`).
- **`_dense_group_ids`** — relabela os ids de grupo (node, ou `(table,col)`
  empacotado) para `[0, S)` por linha, via sort+cumsum+scatter (shapes fixos →
  compatível com `torch.compile`).
- **`_dense_group_ids_global`** — relabela os ids de `(table,col)` para
  `[0, B·S)` sobre **todo o batch** (sort+cumsum+scatter, shapes fixos →
  compatível com `torch.compile`), juntando a mesma feature de todas as
  sequências num **grupo único**. Usado pelo col view quando
  `col_global_pool=True` (default).
- **`_grouped_epps_pulley`** — estatística sketched Epps–Pulley **por grupo**,
  usando `index_add_` num índice 1-D achatado pré-computado pelo chamador
  (`row_flat`/`col_flat`). Células de padding contribuem **zero** para somas e
  contagens (não poluem grupos mesmo em colisão de chave). Grupos com menos de
  `min_group` células válidas são ignorados (default 8). Cada grupo é escalado
  por seu N_g e a perda é a **média ponderada por N_g** (soma das estatísticas
  ÷ total de células válidas, não ÷ nº de grupos) — assim a magnitude **não
  depende do tamanho/quantidade de grupos nem do `batch_size`**, e `λ` fica
  comparável entre col pooled/per-sequence e entre row/col.
- **Pooling do col view** (`col_global_pool`): no paper o column-group são
  **todas** as R linhas de uma feature numa população só. No nanoRT relacional,
  agrupar por sequência dava ~B grupinhos de ~65 células (e descartava ~23% por
  `min_group`). Com pooling global, cada `(table,col)` vira **um** grupo grande e
  uniforme (≈ N·B células) — estimador da função característica muito mais
  estável, que é o que reproduz a redução de variância entre seeds do paper. O
  row view continua por `(sequência, node)` (nós são locais à sequência, não dá
  para juntar entre BFS distintos).
- **`SIGReg`** — orquestra as duas views; amostra direções frescas a cada passo;
  tudo em **fp32**.

### `rt/model.py`
- Novos kwargs no `__init__` (`sigreg_layer`, `sigreg_n_proj`, dims do projector,
  `knots`, `t_max`, `min_group`). `sigreg_layer=None` ⇒ desabilitado e
  **idêntico ao baseline**.
- No `forward`: captura a saída do bloco `sigreg_layer`; computa SIGReg **apenas
  quando `self.training`** (no eval é pulado → **custo zero de inferência**);
  retorna a 3-tupla `(loss, yhat_dict, stats)`. `stats` contém `sigreg_row`,
  `sigreg_col` e os contadores `sigreg_{row,col}/...`; o hidden state não é
  exposto nem armazenado pelo caminho de treino.

### `rt/main.py`
- **Ilha fp32**: após `net.to(torch.bfloat16)`, faz `net.sigreg.float()` (a
  quadratura da CF e o BatchNorm são numericamente sensíveis).
- **Otimizador separado**: `opt_sigreg = AdamW(..., fused=False)` para o
  projector fp32 (evita misturar dtype no AdamW *fused* dos params bf16). Os
  params `sigreg.*` são excluídos do Muon/AdamW principais.
- Combina a perda: `L = (1−λc−λr)·L_sup + λc·R_col + λr·R_row` (§2.4 do paper) e
  loga as métricas (ver §4).

### `rt/optim/__init__.py`
- `build_muon_param_groups` ignora `sigreg.*` (tratado pelo otimizador separado).

### `scripts/nanort_train.py`
- Configuração somente por variáveis `NANORT_SIGREG_*`, centralizada em
  `TrainConfig.from_env()`. A receita do recorde usa layer 0, column view global
  e `λ_col=1e-3`; para desabilitar, usar simultaneamente
  `NANORT_SIGREG_LAYER=-1` e lambdas zero.

### `plx_nanort_train.yaml`
- Inputs `sigreg_layer`, `sigreg_lambda_row`, `sigreg_lambda_col`,
  `sigreg_n_proj` repassados ao script.

---

## 4. Métricas no W&B — o que cada uma significa

**A cada passo:**

| Métrica | Significado |
|---|---|
| `loss` | Perda supervisionada (MTP: Huber p/ number/datetime, BCE p/ boolean). É a tarefa em si. |
| `grad_norm`, `lr_muon`/`lr_adamw` (ou `lr`) | Já existentes. |

**Nos steps de diagnóstico de treino** (1–5 e depois a cada 10 steps):

| Métrica | Significado |
|---|---|
| `train/sigreg_row` | Estatística Epps–Pulley projetada da row view (por `node`). Menor = projeções mais próximas de `N(0,1)`. |
| `train/sigreg_col` | Idem para a column view (por `(table,col)`). |
| `train/sigreg_{row,col}/groups_valid`, `groups_valid_frac` | Quantos grupos passam `min_group` e a fração dos grupos presentes. |
| `train/sigreg_{row,col}/group_size_valid_mean`, `group_size_valid_max` | Tamanho médio/máximo dos grupos válidos. |
| `train/sigreg_{row,col}/cells_covered_frac` | Fração das células válidas coberta pelos grupos aceitos. |

`effective_rank` e `mean_dim_std` são análises desejáveis para o paper, mas **não
são emitidas pela implementação atual**. Se adicionadas, devem rodar fora da
região cronometrada ou ter seu custo contabilizado e ser testadas separadamente.

**Métricas de avaliação (já existentes, são o que decide o speedrun):**

| Métrica | Significado |
|---|---|
| `val/auroc_pooled` (`val/auroc_zero_shot` legado) | AUROC pooled por instância no artefato de val fixo (`rel-avito`). Alvo: **≥ 0.81621**. |
| `train/time_to_target` | Tempo (s) até atingir o alvo de AUROC. **A métrica que o speedrun otimiza** (−1 se não atingiu). |
| `train/elapsed_seconds` | Tempo ativo de treino (budget de 45 min). |
| `val/mtp_loss` | Perda MTP no val. |

### Como ler os experimentos
- **SIGReg está atuando?** `train/sigreg_row` e `train/sigreg_col` devem cair ao
  longo do treino; comparar também cobertura e tamanho de grupos contra o
  baseline explicitamente desabilitado.
- **SIGReg ajuda o speedrun?** Compare `train/time_to_target` (e a curva
  `val/auroc_zero_shot`) contra o baseline. O paper prevê convergência mais
  rápida; confirme que o overhead por passo não anula o ganho.
- **Sweep recomendado** (Fig. 2 do paper): `λc, λr ∈ {0, 0.01, 0.1}` na camada 0.
  No nanoRT o "early" pode diferir (12 blocos vs. ~3), então vale varrer também
  `sigreg_layer ∈ {0, 1, 2}`.

---

## 5. Decisões técnicas e riscos

- **fp32 / bf16**: o modelo roda bf16; o submódulo `sigreg` é mantido em fp32.
- **torch.compile**: `dense_group_ids`/`index_add_` têm shapes fixos e devem
  compilar; `self.training` cria grafos distintos de treino e eval, dentro do
  `cache_size_limit=16`.
- **Memória**: com `batch_size=256` e `n_proj=64`, o SIGReg materializa tensores
  `(B·S, K·17)` ≈ alguns GB. Se ocorrer OOM no H200, **reduza `sigreg_n_proj`**
  (ex.: 32) ou aumente `sigreg_min_group`.
- **DDP não suportado** no caminho SIGReg (falta o all-reduce das cos/sin entre
  ranks). O record é 1×H200, então ok.
- **Checkpoints**: o `state_dict` passa a incluir `sigreg.*`. Carregar um ckpt
  baseline num modelo com SIGReg falharia — não é o caso do speedrun (treina do
  zero, `load_ckpt_path=None`).
- **Regras da competição**: SIGReg é uma mudança de arquitetura + objetivo de
  treino, permitida (não toca val artifact, mix de treino, nem usa `rel-avito`).
  Treino-only ⇒ não altera o protocolo de avaliação zero-shot.

---

## 6. Como rodar no Polyaxon

> Ajuste `hf_artifacts_repo` para o seu repo de artefatos. O `-u` faz upload do
> diretório local (vira `{{ globals.run_artifacts_path }}/uploads`) e
> `hf_token=$HF_TOKEN` acelera o download dos artefatos do HuggingFace.

**Run com SIGReg (camada 0, ambas as views, λ=0.01 — estilo paper):**

```bash
polyaxon run -f plx_nanort_train.yaml -u \
  -P holdout_db=rel-avito \
  -P hf_artifacts_repo=gabrielveloso/nanort-artifacts \
  -P optimizer=muon \
  -P muon_lr=0.005 \
  -P sigreg_layer=0 \
  -P sigreg_lambda_row=0.01 \
  -P sigreg_lambda_col=0.01 \
  -P sigreg_n_proj=64 \
  -P hf_token=$HF_TOKEN
```

**Baseline (SIGReg explicitamente desligado):**

```bash
polyaxon run -f plx_nanort_train.yaml -u \
  -P holdout_db=rel-avito \
  -P hf_artifacts_repo=gabrielveloso/nanort-artifacts \
  -P optimizer=muon \
  -P muon_lr=0.005 \
  -P sigreg_layer=-1 \
  -P sigreg_lambda_row=0 \
  -P sigreg_lambda_col=0 \
  -P hf_token=$HF_TOKEN
```

Para comparação estatística entre seeds (como no paper), rode múltiplos seeds via
`-P seed=0`, `-P seed=1`, …

---

## 7. Referência de hiperparâmetros

| Parâmetro (script / yaml) | Default | Descrição |
|---|---|---|
| `sigreg_layer` | `-1` (off) | Bloco onde aplicar SIGReg (0 = primeiro). |
| `sigreg_lambda_row` (`λr`) | `0.0` | Coef. da row view (por `node`). |
| `sigreg_lambda_col` (`λc`) | `0.0` | Coef. da column view (por `(table,col)`). |
| `sigreg_n_proj` (`K`) | `64` | Nº de projeções 1-D por view (↑ estima melhor, ↑ memória/tempo). |
| `sigreg_lr` | `= lr` | LR do projector (AdamW separado, fp32). |
| `sigreg_proj_hidden_mult` | `4` | `E → mult·E` no projector. |
| `sigreg_proj_out_div` | `4` | Dim de saída do projector = `E/div`. |
| `sigreg_proj_hidden_layers` | `1` | Camadas ocultas do projector (paper: 1 ou 2). |
| `sigreg_knots` | `17` | Nós da quadratura Epps–Pulley. |
| `sigreg_t_max` | `3.0` | Limite de integração `t`. |
| `sigreg_min_group` | `8` | Mín. de células válidas por grupo para entrar na média. |
| `sigreg_col_global_pool` (`--sigreg-col-pool`) | `True` (`1`) | Col view: `1` junta cada feature `(table,col)` num grupo global no batch (recomendado); `0` = grupos por sequência (legacy). |
