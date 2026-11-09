"""Record the implementation repair before any valid v24 model evaluation."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
DEST=ROOT.parent/"entregaveis/v24"
directory=ROOT/"v24"
old=directory/"attempt_02_scaling_error"
assert (old/"invalidation.json").exists()
path=directory/"gci_home_credit_v24_pesquisa.ipynb"
assert hashlib.sha256(path.read_bytes()).hexdigest()=="e0d0e467e0c69fda6c58034dcea79f0afe09a8a887c71fd7403e6c2fdb45150a"
nb=json.loads(path.read_text())
nb["cells"][26]["source"]=(ROOT/"v24_fm.py").read_text()
correction="""

## Correção de implementação, congelada antes da triagem válida

A primeira implementação confundia qualquer coluna com dois valores distintos
com um indicador 0/1. NAME_CONTRACT_TYPE_COUNT e CODE_GENDER_COUNT ficaram em
escala de dezenas/centenas de milhares, contrariando o propósito da normalização
numérica. A tentativa pesquisa_02 foi interrompida antes da seleção de peso;
seus resultados FM são inválidos para avaliar a hipótese e foram preservados
em evidence/attempt_02_scaling_error. Não foi executada confirmação.

Correção única: somente valores observados contidos em {0,1} são indicadores;
as demais numéricas de dois níveis também passam pelos quantis. Nenhuma mudança
de arquitetura, seed, learning rate, regularização, épocas máximas, grade,
estratos ou critério de promoção. Protocolo revisão 2 congelado antes da nova
tentativa pesquisa_03. O teste inclui contagens de dois níveis de grande escala.
"""
nb["cells"][0]["source"]+=correction
for cell in nb["cells"]:
    if cell["cell_type"]=="code":cell.update(execution_count=None,outputs=[])
original=json.loads((ROOT/"v17_definitions_sha256.json").read_text())
found={}
for cell in nb["cells"]:
    if cell["cell_type"]!="code":continue
    source="".join(cell["source"])
    for node in ast.parse(source).body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            found[node.name]=hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest()
assert all(found[k]==v for k,v in original.items())
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n")
environment={"np":__import__("numpy")}
for node in ast.parse((ROOT/"v24_fm.py").read_text()).body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {"V24_WEIGHTS","V24_PROTOCOL"} for t in node.targets):
        exec(compile(ast.Module(body=[node],type_ignores=[]),"protocol","exec"),environment)
for target in [directory/"evidence",DEST/"evidence"]:
    target.mkdir(exist_ok=True)
    (target/"frozen_protocol.json").write_text(json.dumps(environment["V24_PROTOCOL"],ensure_ascii=False,indent=2)+"\n")
for name in ["v24_fm.py","check_v24.py","repair_v24_preprocessing.py"]:
    shutil.copy2(ROOT/name,DEST/name)
shutil.copy2(path,DEST/path.name)
(DEST/"PLANO.md").write_text(nb["cells"][0]["source"])
audit=json.loads((old/"source_audit.json").read_text())
audit.update(research_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    protocol_sha256=hashlib.sha256((directory/"evidence/frozen_protocol.json").read_bytes()).hexdigest(),
    fm_module_sha256=hashlib.sha256((ROOT/"v24_fm.py").read_bytes()).hexdigest(),
    implementation_revision=2,previous_attempt_invalid=True,
    implementation_repair="normalize two-level counts; only true 0/1 flags skip quantiles",
    frozen_before_valid_training=True)
(DEST/"evidence/source_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
print(json.dumps(audit,indent=2))
