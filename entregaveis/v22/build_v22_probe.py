"""Self-contained feature probe, keeping all original v17 definitions."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
nb=json.loads((ROOT/"gci_home_credit_v21.ipynb").read_text())
cells=nb["cells"][:25]
cells[0]=dict(cell_type="markdown",metadata={},source=(
    "# v22 — triagem de segmentos de densidade do empréstimo\n\n"
    "Adaptação própria da ideia relatada pelo usuário no Slack, atribuída a Theophile Avenel e sameenakb. "
    "Ablation apenas de LightGBM B seed42 nas partes originais0–1: baseline, KDE global, KDE por contrato. "
    "Densidade aproximada por histograma suavizado de log(crédito/prestação), 2048bins, bandwidth Silverman, "
    "prominência1% e distância12bins. Picos/mínimos geram sete features; ajustados só nas linhas de treino da parte, semTARGET/ID. "
    "Os pré-processadores originais continuam preservados, incluindo estatísticas não supervisionadas train+test. "
    "Isso não demonstra uma versão sem transdução. O experimento não escreve CSV, nem valida o ensemble completo. "
    "Uma configuração só avança se melhorar AUC conjunta0–1 e a média das diferenças por parte. "
    "Ainda requer seleção de pesos0–1, confirmação2–4 e total, treino integral e dupla reprodução antes de submissão.\n\n"
    "Fonte da ideia, relato do usuário: https://gciworld2026sept3.slack.com/archives/C0BDW0TT2HZ/p1792965969389949 . "
    "Não foi possível auditar o código antigo completo de pricing; registros disponíveis descrevem TE de15chaves, semKDE.\n"))
cells.append(dict(cell_type="code",metadata={},execution_count=None,outputs=[],source=(ROOT/"v22_density.py").read_text()))
cells.append(dict(cell_type="code",metadata={},execution_count=None,outputs=[],source='''
started=time.time()
train=pd.read_csv(BASE_DIR/"input/train.csv")
test=pd.read_csv(BASE_DIR/"input/test.csv")
assert train.shape==(171202,34) and test.shape==(61500,33)
input_hashes={name:hashlib.sha256((BASE_DIR/"input"/name).read_bytes()).hexdigest() for name in ["train.csv","test.csv"]}
ctx=prepare(train,test)
report=v22_probe(ctx,BASE_DIR/"evidence")
report["seconds_including_prepare"]=time.time()-started
execution_report=dict(report,run_cv=True,cv_only=True,csv_sha256=None)
(BASE_DIR/"evidence/probe_report.json").write_text(json.dumps(execution_report,ensure_ascii=False,indent=2)+"\\n")
'''))
found={}
for c in cells:
    if c["cell_type"]!="code":continue
    c.update(execution_count=None,outputs=[])
    s="".join(c["source"])
    for n in ast.parse(s).body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)):
            assert n.name not in found
            found[n.name]=hashlib.sha256(ast.get_source_segment(s,n).encode()).hexdigest()
original=json.loads((ROOT/"v17_definitions_sha256.json").read_text())
assert all(found[k]==v for k,v in original.items())
nb["cells"]=cells
p=ROOT/"v22";p.mkdir(exist_ok=True)
(p/"gci_home_credit_v22_pesquisa.ipynb").write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n")
if not (p/"input").exists():(p/"input").symlink_to(ROOT/"input",target_is_directory=True)
print(p,"24 original definitions preserved")
