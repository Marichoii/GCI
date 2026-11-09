"""Build a self-contained v23 log-odds blend validation notebook."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
from organizar_entregaveis import organize

ROOT = Path(__file__).resolve().parent
assert not (ROOT.parent/"entregaveis/v23/rank.json").exists(), "Do not overwrite existing v23 research or history."
nb = json.loads((ROOT/"gci_home_credit_v21.ipynb").read_text())
nb["cells"] = nb["cells"][:26]
nb["cells"][0]["source"] = (
    "# v23 — validação de ensemble em log-odds\n\n"
    "Todos os sete componentes da v21 treinados novamente usando apenas input/train.csv e input/test.csv. "
    "Sem carregar previsões, OOFs, modelos ou checkpoints. Pesos e sementes da v21 fixos; duas threads por modelo. "
    "Comparar percentis ao quadrado ao rank da média ponderada dos log-odds (clip 1e-6). "
    "Selecionar apenas alpha=0,0.1,...,1 nas partes0–1, preservando pesos dos componentes. "
    "Se não houver ganho0–1, parar. Caso contrário, completar OOF, refixar alpha apenas0–1 "
    "com ranks globais e congelar antes de medir2–4 e total. Somente ganhos em ambos permitem treino integral. "
    "Este notebook de pesquisa não escreve CSV. As funções originais da v17, inclusive pré-processamento "
    "não supervisionado train+test, são preservadas. Os log-odds dos modelos não são necessariamente calibrados.\n\n"
    "Fonte da hipótese: resumo do Slack fornecido pelo usuário, https://gciworld2026sept3.slack.com/archives/C0BDW0TT2HZ/p1792965992208839 . "
    "Não há garantia de reproduzir o ganho informado por outro participante.\n")
nb["cells"].append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=(ROOT/"v23_logodds.py").read_text()))
nb["cells"].append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source='''
started=time.time()
train=pd.read_csv(BASE_DIR/"input/train.csv")
test=pd.read_csv(BASE_DIR/"input/test.csv")
assert train.shape==(171202,34) and test.shape==(61500,33)
input_hashes={name:hashlib.sha256((BASE_DIR/"input"/name).read_bytes()).hexdigest() for name in ["train.csv","test.csv"]}
assert input_hashes=={"train.csv":"b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1", "test.csv":"dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65"}
ctx=prepare(train,test)
add_tabm_feature_set(ctx)
report=v23_run(ctx,BASE_DIR/"evidence")
report["seconds_including_prepare"]=time.time()-started
execution_report=dict(report,run_cv=True,cv_only=True,csv_sha256=None)
(BASE_DIR/"evidence/experiment_report.json").write_text(json.dumps(execution_report,ensure_ascii=False,indent=2)+"\\n")
'''))
found = {}
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    c.update(execution_count=None, outputs=[])
    s = "".join(c["source"])
    for node in ast.parse(s).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            assert node.name not in found
            found[node.name] = hashlib.sha256(ast.get_source_segment(s, node).encode()).hexdigest()
original = json.loads((ROOT/"v17_definitions_sha256.json").read_text())
assert all(found[k] == v for k, v in original.items())
directory = ROOT/"v23"
directory.mkdir(exist_ok=True)
path = directory/"gci_home_credit_v23_pesquisa.ipynb"
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1)+"\n")
if not (directory/"input").exists():
    (directory/"input").symlink_to(ROOT/"input", target_is_directory=True)
dest = ROOT.parent/"entregaveis/v23"
(dest/"evidence").mkdir(parents=True, exist_ok=True)
shutil.copy2(path, dest/path.name)
for name in ["v23_logodds.py", "build_v23.py", "execute_notebook.py", "v17_definitions_sha256.json"]:
    shutil.copy2(ROOT/name, dest/name)
shutil.copy2(ROOT.parent/"entregaveis/v21/requirements.txt", dest/"requirements.txt")
(dest/"PLANO.md").write_text(nb["cells"][0]["source"]+"\nPlano fixado em 30/10/2026, antes dos resultados. Os folds já foram consultados em pesquisas anteriores; esta confirmação não é um holdout intocado.\n")
(dest/"evidence/source_audit.json").write_text(json.dumps(dict(original_definitions_verified=len(original), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), threads=2, cv_workers=2, selection_folds=[0,1], confirmation_folds=[2,3,4], seed=42, full_reproduction_not_executed=True), indent=2)+"\n")
(dest/"evidence/experiment_report.json").write_text(json.dumps(dict(status="running_selection", confirmed=False, csv_sha256=None), indent=2)+"\n")
rank_path = dest/"rank.json"
assert not rank_path.exists(), "Do not overwrite existing v23 run or history."
rank_path.write_text(json.dumps(dict(versao="v23", historico=[], score_publico=None, rank_publico=None, participantes=None, data_relato=None, fonte="Pesquisa local iniciada em 30/10/2026; sem submissão v23", observacao="Validação da combinação em log-odds; nenhum CSV aprovado.", notion_url=None), ensure_ascii=False, indent=2)+"\n")
organize("v23")
print(path, "24 original definitions preserved")
