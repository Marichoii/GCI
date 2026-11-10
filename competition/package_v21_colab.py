"""Deliver a v21 operational Colab copy; preserve historical notebook/ZIP/CSV."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from organizar_entregaveis import organize

ROOT=Path(__file__).resolve().parent
DEST=ROOT.parent/"entregaveis/v21/colab"
PARTS=ROOT/"v21_colab"
tests=json.loads((PARTS/"evidence/test_report.json").read_text())
audit=json.loads((PARTS/"evidence/model_preservation.json").read_text())
assert tests["passed"] and not tests["training_executed"] and not tests["colab_executed"]
assert audit["original_full_training_cell_unchanged"] and audit["frozen_v21_unchanged"]
assert audit["colab_notebook_sha256"]==tests["notebook_sha256"]
for name,digest in tests["historical_artifacts_unchanged"].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
for name in ["gci_home_credit_v21_colab.ipynb","COMO_RODAR_NO_COLAB.md"]:
    shutil.copy2(ROOT/name,DEST/name)
shutil.copy2(PARTS/"v21_colab.diff",DEST/"v21_colab.diff")
shutil.copytree(PARTS/"evidence",DEST/"evidence",dirs_exist_ok=True)
(DEST/"input").mkdir(exist_ok=True)
shutil.copy2(ROOT/"input/README.md",DEST/"input/README.md")
sources={
    "colab_python_announcement":"https://github.com/googlecolab/colabtools/issues/6081",
    "runtime_versions":"https://research.google.com/colaboratory/runtime-version-faq.html",
    "torch_cpu":"https://download.pytorch.org/whl/cpu/torch/",
    "pip_subprocess":"https://pip.pypa.io/en/stable/user_guide/#using-pip-from-your-program",
    "ipython_cell_events":"https://ipython.readthedocs.io/en/stable/config/callbacks.html",
    "colab_io":"https://colab.research.google.com/notebooks/io.ipynb",
    "colab_limits":"https://research.google.com/colaboratory/faq.html"}
(DEST/"evidence/sources.json").write_text(json.dumps(dict(checked_on="2026-11-10",sources=sources),indent=2)+"\n")
summary=f"""# v21 — cópia operacional para Colab

Preparação de 10/11/2026, sem nova versão de modelagem.

- [Notebook](gci_home_credit_v21_colab.ipynb)
- [Passo a passo](COMO_RODAR_NO_COLAB.md)
- [Diff legível das células](v21_colab.diff)
- [Auditoria de preservação](evidence/model_preservation.json)
- [Testes](evidence/test_report.json)
- [Wheels oficiais auditados](evidence/wheel_metadata.json)

Somente o cabeçalho e a célula original de ambiente mudaram. As outras
26 células originais foram preservadas integralmente, inclusive a célula
completa de treino e FROZEN_V21. Há adições de ambiente/entrada, cronometragem,
verificação e exportação. O diff compara o texto extraído das células, com
marcadores legíveis; não é um patch do JSON do notebook.

O notebook original validado, seu ZIP e seu CSV não foram substituídos.
Notebook original SHA-256: `{audit['original_notebook_sha256']}`.
Cópia Colab SHA-256: `{audit['colab_notebook_sha256']}`.
FROZEN_V21 SHA-256 canônico: `{audit['frozen_v21_sha256']}`.

Testes locais: {tests['tests_run']} passaram. Foram verificadas as guardas de
reinício/instalação, hashes, upload/Drive com dublês, tempos/eventos de célula
sem executar o treino, CSV histórico e casos negativos. Foram resolvidas as
dependências Linux/Python 3.13 e 3.12 só com wheels, sem instalar pacotes.

Não executados: instalação real no Colab, transferências autenticadas
upload/Drive, treino novo, reprodução Linux/Python 3.13, benchmark de 90 min e
desconexão real. Nenhum CSV de submissão novo foi gerado. Os logs de teste
não são evidência de treinamento. A referência de CSV macOS continua com
SHA-256 `48334efc455a9a28f395e945fb4c04ee847458f439d8b26984e3965766d6a756`.

O arquivo requirements.txt é idêntico ao da v21. Os arquivos
resolved_linux_cp*.txt são auditorias de resolução e não substituem os pins.
O ZIP de entrega não contém dados, previsões, OOFs, modelos ou checkpoints.
"""
(DEST/"README.md").write_text(summary)
archive=DEST/"gci_home_credit_v21_colab.zip"
with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as z:
    for path in sorted(DEST.rglob("*")):
        if path.is_file() and path not in {archive,DEST/"manifesto_sha256.json"}:
            z.write(path,str(path.relative_to(DEST)))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert not any(name.endswith(("train.csv","test.csv","submission.csv",".npz",".pt",".pth")) for name in z.namelist())
    assert z.read("gci_home_credit_v21_colab.ipynb")== (ROOT/"gci_home_credit_v21_colab.ipynb").read_bytes()
manifest={str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(DEST.rglob("*")) if p.is_file() and p.name!="manifesto_sha256.json"}
(DEST/"manifesto_sha256.json").write_text(json.dumps(manifest,indent=2)+"\n")
organize("v21")
for name,digest in tests["historical_artifacts_unchanged"].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
print(json.dumps(dict(notebook= str(DEST/"gci_home_credit_v21_colab.ipynb"),
    notebook_sha256=audit["colab_notebook_sha256"],archive_bytes=archive.stat().st_size,
    archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    historical_artifacts_unchanged=True,no_training=True),indent=2))
