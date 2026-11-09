"""Freeze the v24 protocol and self-contained research notebook before fitting."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
from organizar_entregaveis import organize

ROOT = Path(__file__).resolve().parent
DEST = ROOT.parent/"entregaveis/v24"
assert not DEST.exists(), "Preserve existing v24 artifacts."
nb = json.loads((ROOT/"gci_home_credit_v21.ipynb").read_text())
nb["cells"] = nb["cells"][:26]
plan = """# v24 — Factorization Machine PyTorch

Protocolo fixado em 09/11/2026 antes dos treinos; pesquisa termina em 14/11/2026.
Usar somente input/train.csv e input/test.csv. Nenhuma biblioteca nova.
Sem carregar previsões, OOFs, checkpoints ou modelos. Sem SK_ID_CURR como
variável, sem transformações de ID e sem regras manuais nas previsões.

FM de segunda ordem: termo linear e fatores de posto 16, perda BCE sem pesos,
AdamW lr=0,002, weight_decay=0,001, batch=2048, seed=42 e duas threads.
Variáveis plusx da v21, categorias one-hot; numéricas contínuas com quantis
normais (256 quantis), clip [-4,4], ausentes=0 com indicadores. Ajustar tudo
somente no treino correspondente. Inicialização dos fatores N(0,0,01), termo
linear zero e intercepto logit da prevalência do treino.

Para cada parte externa, reservar 15% de seu treino (estratificado, seed 42)
para escolher épocas: até 40, patience 6, min_delta 1e-6 por AUC. Recalcular
features supervisionadas dentro desse treino interno. Recomeçar do zero no
treino externo completo com as épocas escolhidas. Épocas finais = média
arredondada das épocas internas, sem multiplicador. Nenhum rótulo da parte
externa escolhe épocas da FM. Os sete componentes v21 e seu early stopping
original são preservados para reproduzir a referência; seu pré-processamento
não supervisionado herdado usa covariáveis train+test.

Cinco partes StratifiedKFold(5,shuffle=True,random_state=42), ordem original.
Estratos exclusivamente diagnósticos: chave round(crédito/prestação,2),
frequência contada apenas no treino externo; 0–9 (raras, inclui inéditas),
10–49, 50–199, >=200; ausentes/inválidas separados (crédito/prestação não
positivos ou não finitos). Congelar atribuição a estratos antes do primeiro
fit. Não usar estratos para features, pesos de treino ou regras de previsão.
Reportar linhas, %, positivos, AUC v21, FM e ensemble candidato por estrato.

Grade fixa de peso FM [0%,5%,10%,15%,20%], preservando proporções v21;
empates preferem menor peso. Percentis ao quadrado, como v21/v18. Selecionar
somente em 0–1. Corte rápido com percentis das OOF disponíveis: se peso zero
ou sem ganho conjunto 0–1, parar sem treinar 2–4 ou gerar CSV. Se passar,
completar OOF e refixar peso na mesma grade usando somente rótulos 0–1 e
percentis globais; congelar antes de calcular AUCs 2–4/total. Sem retuning.
Reportar AUC FM isolada e Spearman com cada um dos sete componentes.

Promoção exige delta 2–4 >=0,00010 e delta total >0 sobre v21 reproduzida
(2–4=0,7747809092432562; total=0,7740567649584406). Estratos não acrescentam
um critério de promoção; servem para testar a justificativa. Se falhar,
nenhum CSV. Se passar, notebook final treina tudo em 100% do treino;
duas execuções novas, mesmo SHA-256 e menos de 60 min por execução completa
em CPU. Tempos de Colab precisam de verificação; macOS não prova identidade
entre plataformas. Esta pesquisa usa dois workers de CV, duas threads cada;
o treino final será sequencial com duas threads.

Este notebook de pesquisa não escreve CSV. Somente após aprovação será
preparado o notebook final com receita congelada. Os folds já foram usados
em pesquisas anteriores e não são um holdout intocado.

Referência do mecanismo: https://libfm.org/ . Não são usados dados externos.
"""
nb["cells"][0]["source"] = plan
nb["cells"].append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=(ROOT/"v24_fm.py").read_text()))
nb["cells"].append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source='''
started=time.time()
evidence_dir=BASE_DIR/"evidence"
protocol_sha256=v24_frozen_protocol(evidence_dir)
train=pd.read_csv(BASE_DIR/"input/train.csv")
test=pd.read_csv(BASE_DIR/"input/test.csv")
assert train.shape==(171202,34) and test.shape==(61500,33)
input_hashes={name:hashlib.sha256((BASE_DIR/"input"/name).read_bytes()).hexdigest() for name in ["train.csv","test.csv"]}
assert input_hashes=={"train.csv":"b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1", "test.csv":"dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65"}
# Freeze diagnostic strata before prepare(), which fits auxiliary regressors.
frozen_splits=list(StratifiedKFold(**V24_PROTOCOL["folds"]).split(np.zeros(len(train)),train.TARGET))
frozen_labels,frozen_frequency=v24_strata(train,frozen_splits)
frozen_ids=np.empty(len(train),dtype=np.int8)
for f,(_,b) in enumerate(frozen_splits):frozen_ids[b]=f
pd.DataFrame(dict(row=np.arange(len(train)),fold=frozen_ids,outer_train_frequency=frozen_frequency,stratum=frozen_labels)).to_csv(evidence_dir/"frozen_strata.csv",index=False)
print("Protocolo e estratos congelados antes de prepare/fit",protocol_sha256,flush=True)
print(pd.Series(frozen_labels).value_counts().to_string(),flush=True)
ctx=prepare(train,test)
add_tabm_feature_set(ctx)
report=v24_run(ctx,train,evidence_dir)
report["seconds_including_prepare"]=time.time()-started
execution_report=dict(report,run_cv=True,cv_only=True,csv_sha256=None)
v24_json(evidence_dir/"experiment_report.json",execution_report)
'''))
nb["cells"].append(dict(cell_type="markdown", metadata={}, source="Pesquisa v24: evidências de execução ficam em evidence/. Nenhuma submissão automática."))
found = {}
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    cell.update(execution_count=None, outputs=[])
    source = "".join(cell["source"])
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            assert node.name not in found
            found[node.name] = hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()
original = json.loads((ROOT/"v17_definitions_sha256.json").read_text())
assert all(found[k] == v for k,v in original.items())
directory = ROOT/"v24"
directory.mkdir(exist_ok=False)
path = directory/"gci_home_credit_v24_pesquisa.ipynb"
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1)+"\n")
(directory/"input").symlink_to(ROOT/"input", target_is_directory=True)
(DEST/"evidence").mkdir(parents=True)
shutil.copy2(path, DEST/path.name)
for name in ["v24_fm.py", "build_v24.py", "execute_notebook.py", "v17_definitions_sha256.json"]:
    shutil.copy2(ROOT/name, DEST/name)
shutil.copy2(ROOT.parent/"entregaveis/v21/requirements.txt", DEST/"requirements.txt")
(DEST/"PLANO.md").write_text(plan)
# Obtain the protocol literal without importing or fitting any learner.
module = ast.parse((ROOT/"v24_fm.py").read_text())
environment = {"np":__import__("numpy")}
for node in module.body:
    if isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id in {"V24_WEIGHTS","V24_PROTOCOL"} for t in node.targets):
        exec(compile(ast.Module(body=[node], type_ignores=[]), "protocol", "exec"), environment)
protocol = environment["V24_PROTOCOL"]
for target in [DEST/"evidence", directory/"evidence"]:
    target.mkdir(exist_ok=True)
    (target/"frozen_protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+"\n")
(DEST/"evidence/source_audit.json").write_text(json.dumps(dict(original_definitions_verified=len(original),
    research_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    protocol_sha256=hashlib.sha256((directory/"evidence/frozen_protocol.json").read_bytes()).hexdigest(),
    fm_module_sha256=hashlib.sha256((ROOT/"v24_fm.py").read_bytes()).hexdigest(),
    frozen_before_training=True, cv_workers=2, threads_per_model=2,
    full_reproduction_not_executed=True, no_new_dependencies=True),indent=2)+"\n")
(DEST/"evidence/experiment_report.json").write_text(json.dumps(dict(status="running_selection",confirmed=False,csv_sha256=None),indent=2)+"\n")
(DEST/"rank.json").write_text(json.dumps(dict(versao="v24",historico=[],score_publico=None,rank_publico=None,
    participantes=None,data_relato=None,fonte="Pesquisa local de 09/11/2026; sem submissão v24",
    observacao="FM PyTorch, receita e estratos congelados; corte rápido 0–1; sem CSV aprovado.",notion_url=None),ensure_ascii=False,indent=2)+"\n")
organize("v24")
print(path, "24 original definitions preserved; protocol frozen before training")
