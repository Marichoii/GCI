"""Environment-only adaptation of v21; original model cells stay byte-identical."""
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/"entregaveis/v21/gci_home_credit_v21.ipynb"
PARTS=ROOT/"v21_colab"
DEST=ROOT.parent/"entregaveis/v21/colab"
DEST.mkdir(parents=True,exist_ok=True)
old=json.loads(SOURCE.read_text())
new=copy.deepcopy(old)
new["metadata"]["language_info"]["version"]="3.13"

def code(name):
    source=(PARTS/(name+".py")).read_text()
    if name=="antes_do_treino":
        source=source.replace("__ORIGINAL_MAIN_SHA256__",hashlib.sha256("".join(old["cells"][26]["source"]).strip().encode()).hexdigest())
    return dict(cell_type="code",id="v21-colab-"+name,metadata={"v21_colab_role":name},execution_count=None,outputs=[],source=source)

def md(text):
    return dict(cell_type="markdown",id="v21-colab-doc-"+hashlib.sha256(text.encode()).hexdigest()[:12],metadata={"v21_colab_role":"documentacao"},source=text)

new["cells"][0]["source"]="""# Home Credit v21 — execução no Google Colab CPU

Mesma receita congelada da v21. Nenhum modelo, feature, seed, peso, tamanho ou
FROZEN_V21 foi alterado. Pesquisa encerrada; executar apenas o treino integral.

Primeiro execute somente a célula de ambiente. Se instalar pacotes, ela para:
**reinicie o ambiente e rode tudo de novo**. Depois do reinício, use Executar tudo.
Selecione upload ou Drive na célula de entrada; arquivos existentes corretos
são reutilizados. Não são carregadas previsões, OOFs, checkpoints ou modelos.

As células originais de modelagem e a célula original de treino integral foram
preservadas integralmente. As adições são ambiente, entrada, cronometragem,
verificação e exportação. Consulte COMO_RODAR_NO_COLAB.md e o diff anexo.
"""
new["cells"][1]=code("ambiente")
new["cells"]=new["cells"][:2]+[
    md("## Entrada opcional: upload ou Google Drive\n\nSó os dois CSVs oficiais do GCI; SHA-256 conferido antes de qualquer treino. Se input/ já estiver correto, esta célula apenas o valida."),
    code("entrada")]+new["cells"][2:]
main_index=next(i for i,c in enumerate(new["cells"]) if c.get("source")==old["cells"][26]["source"])
new["cells"][main_index:main_index]=[
    md("## Ambiente/entrada: conferir antes do treino\n\nO bloco seguinte mede o início da etapa e impede pesquisa ou sobrescrita acidental. A célula de treino logo depois dele é idêntica à original."),code("antes_do_treino")]
main_index+=2
new["cells"][main_index+1:main_index+1]=[
    md("## Verificação final e tempos\n\nReabre o CSV gravado, confere formato, IDs, NaN, limites e SHA-256. O aviso de 90 minutos é emitido ao concluir a célula original de treino; não muda nem cancela modelos."),
    code("verificacao"),
    md("## Guardar resultados — opcional\n\nBaixa um ZIP com CSV, hashes e relatórios; a cópia para Drive é opcional e usa uma pasta nova. Nenhum modelo ou checkpoint é salvo."),
    code("guardar_resultados")]
original_mapping=[]
for i,cell in enumerate(old["cells"]):
    if i in [0,1]:continue
    matches=[j for j,c in enumerate(new["cells"]) if c==cell]
    assert len(matches)==1,(i,matches)
    original_mapping.append(dict(original_cell=i,colab_cell=matches[0],unchanged=True,
        source_sha256=hashlib.sha256("".join(cell["source"]).encode()).hexdigest()))
assert new["cells"][main_index]==old["cells"][26]
old_code="\n".join("".join(c["source"]) for c in old["cells"] if c["cell_type"]=="code")
new_code="\n".join("".join(c["source"]) for c in new["cells"] if c["cell_type"]=="code")
def frozen(source):
    return next(ast.literal_eval(node.value) for node in ast.parse(source).body if isinstance(node,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=="FROZEN_V21" for t in node.targets))
assert frozen(old_code)==frozen(new_code)
for c in new["cells"]:
    if c["cell_type"]=="code":compile("".join(c["source"]),"notebook_cell","exec")
data=(json.dumps(new,ensure_ascii=False,indent=1)+"\n").encode()
name="gci_home_credit_v21_colab.ipynb"
(ROOT/name).write_bytes(data)
(DEST/name).write_bytes(data)

def render(nb):
    chunks=[]
    for cell in nb["cells"]:
        label="CODE" if cell["cell_type"]=="code" else "MARKDOWN"
        # No numerical cell indices, so inserted cells do not obscure equality.
        chunks.append(f"\n# ===== {label} =====\n"+"".join(cell["source"])+"\n")
    return "".join(chunks)

diff="".join(difflib.unified_diff(render(old).splitlines(keepends=True),render(new).splitlines(keepends=True),
    fromfile="gci_home_credit_v21.ipynb (células legíveis)",tofile=name+" (células legíveis)",n=3))
(PARTS/"v21_colab.diff").write_text(diff)
(DEST/"v21_colab.diff").write_text(diff)
(PARTS/"evidence").mkdir(exist_ok=True)
audit=dict(original_notebook_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    colab_notebook_sha256=hashlib.sha256(data).hexdigest(),
    frozen_v21_unchanged=True,original_full_training_cell_unchanged=True,
    unchanged_original_cells=original_mapping,changed_original_cells=[0,1],
    added_cell_roles=["documentacao","entrada","antes_do_treino","verificacao","guardar_resultados"],
    training_executed=False,colab_executed=False,
    environment_metadata_change={"language_info.version":{"old":old["metadata"]["language_info"]["version"],"new":"3.13"}},
    frozen_v21_sha256=hashlib.sha256(json.dumps(frozen(old_code),sort_keys=True).encode()).hexdigest())
(PARTS/"evidence/model_preservation.json").write_text(json.dumps(audit,indent=2)+"\n")
for name in ["requirements.txt"]:
    shutil.copy2(SOURCE.parent/name,DEST/name)
print(json.dumps({k:v for k,v in audit.items() if k!="unchanged_original_cells"},indent=2))
