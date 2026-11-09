"""Archive a completed negative experiment, with no submission CSV."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from organizar_entregaveis import organize

ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT / "v19"
DEST = ROOT.parent / "entregaveis/v19"


def package():
    report = json.loads((RESEARCH / "evidence/experiment_report.json").read_text())
    assert report["status"].startswith("rejected_"), "Only package completed negative research here."
    assert report["csv_sha256"] is None and not report["confirmed"]
    evidence = DEST / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    for source, target in [
        (RESEARCH / "gci_home_credit_v19_pesquisa.ipynb", DEST / "gci_home_credit_v19_pesquisa.ipynb"),
        (ROOT / "v19_linear.py", DEST / "v19_linear.py"),
        (ROOT / "build_v19_experiment.py", DEST / "build_v19_experiment.py"),
        (ROOT / "execute_notebook.py", DEST / "execute_notebook.py"),
        (ROOT / "requirements.txt", DEST / "requirements.txt"),
        (RESEARCH / "installed_versions_current.txt", DEST / "installed_versions_local.txt"),
        (ROOT / "v17_definitions_sha256.json", DEST / "v17_definitions_sha256.json"),
        (RESEARCH / "evidence/experiment_report.json", evidence / "experiment_report.json"),
        (RESEARCH / "runs/pesquisa_linear_01/executed.ipynb", evidence / "executed.ipynb"),
        (RESEARCH / "runs/pesquisa_linear_01/execution_report.json", evidence / "execution_report.json"),
        (RESEARCH / "pesquisa.log", evidence / "pesquisa.log"),
    ]:
        shutil.copy2(source, target)
    notebook = (DEST / "gci_home_credit_v19_pesquisa.ipynb").read_bytes()
    assert notebook == (RESEARCH / "runs/pesquisa_linear_01/source.ipynb").read_bytes()
    lines = ["# Resultado da pesquisa v19", "", "**Não houve ganho aprovado; nenhum CSV novo foi gerado.**", ""]
    if report["status"] == "rejected_in_screening":
        lines += ["As duas configurações do LightGBM com folhas lineares foram comparadas ao ensemble v18 nas partes 0–1. "
                  "Nenhuma recebeu peso positivo na seleção. As partes 2–4 não foram executadas, portanto não há uma nova AUC total.", "",
                  "| Configuração | AUC v18, partes 0–1 | Melhor AUC do ensemble | Peso selecionado | Melhor AUC com peso > 0 |",
                  "|---|---:|---:|---:|---:|"]
        for row in report["screening"]:
            positive = max((g for g in row["grid"] if g["alpha"] > 0), key=lambda r:r["auc"])
            lines.append(f'| {row["configuration"]["label"]} | {row["baseline_auc"]:.8f} | {row["best"]["auc"]:.8f} | '
                         f'{row["best"]["alpha"]:.3f} | {positive["auc"]:.8f} (peso {positive["alpha"]:.3f}) |')
        lines += ["", "Os percentis da triagem usam somente as linhas das partes 0–1. Não comparar esta AUC diretamente com "
                  "a AUC das mesmas partes calculada com os percentis globais da validação completa da v18.", ""]
        lines += ["| Parte | v18 recalculada na triagem | v19 com o peso selecionado (zero) |",
                  "|---|---:|---:|"]
        for fold in [0, 1]:
            row = next(row for row in report["screen_table"] if row["parte"] == fold)
            lines.append(f'| {fold} | {row["v18"]:.8f} | {row["v19"]:.8f} |')
        lines += [f'| 0–1 | {report["screening"][0]["baseline_auc"]:.8f} | {report["screening"][0]["baseline_auc"]:.8f} |',
                  "| 2, 3, 4 e total | Não executado nesta pesquisa | Não executado |", "",
                  "O empate significa que a seleção excluiu o novo componente; não é um ganho.", ""]
    else:
        lines += ["Houve sinal na triagem, mas a receita escolhida nas partes 0–1 não melhorou simultaneamente 2–4 e o total.", "",
                  "| Parte | v18 | v19 | Diferença |", "|---|---:|---:|---:|"]
        for row in report["table"]:
            lines.append(f'| {row["parte"]} | {row["v18"]:.8f} | {row["v19"]:.8f} | {row["delta"]:+.8f} |')
    lines += ["", "## O que foi executado", "",
              "- Notebook de pesquisa executado em processo Python novo, todas as células de código em ordem, pelo runner incluído. "
              "A versão com saídas e o log estão em `evidence/`.",
              "- Apenas `input/train.csv` e `input/test.csv`. Todos os modelos usados nesta comparação treinados do zero; "
              "OOFs apenas na memória. Não foi carregado o CSV da v18 para validar.",
              "- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, ordem original. "
              "`Supervised.add` usa o treino da parte e os mesmos cálculos internos da v17. "
              "Trocar os TARGET da validação não alterou os frames do novo componente.",
              "- Funções originais da v17 preservadas; wrapper estrito da MLP mantido. Sementes e duas threads fixas.",
              f'- Tempo efetivo: {report["total_seconds_including_prepare"]/60:.2f} minutos, macOS ARM64 CPU. Não medido no Colab.',
              "- Não houve treino integral de uma nova candidata nem duas reproduções de CSV; não existe um novo SHA-256 de submissão.",
              "- Score público e rank da v19 não existem: nenhuma submissão nova foi feita.", "",
              "## Receita testada", "",
              "LightGBM binário: 9 folhas, profundidade máxima 4, mínimo 500 linhas/folha, lr 0,03, "
              "até 4.000 árvores, early stopping 150, fração de variáveis 0,44, bagging 0,74, "
              "L2 2, `linear_lambda` em {10, 100}, seed 42. Quantis normais 256 ajustados no treino, "
              "imputação pela mediana e indicadores de ausência. Categorias nativas. Sem feature de ID.", "",
              "Pesos da v18 mantidos: 0,19875; 0,03825; 0,2235; 0,2895; 0,25. "
              "Peso do componente linear em 0–30%, passo 2,5%; proporções anteriores preservadas.", "",
              "Fonte técnica: [documentação primária do LightGBM 4.6](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#linear_tree).", "",
              "## Como repetir", "",
              "Coloque o notebook ao lado de `input/train.csv` e `input/test.csv` e execute todas as células. "
              "As versões necessárias são instaladas/verificadas na primeira célula. "
              "Para usar o runner, no ambiente com `requirements.txt` instalado:", "", "```sh",
              "python execute_notebook.py --notebook gci_home_credit_v19_pesquisa.ipynb --name pesquisa_linear_01", "```", "",
              "O diretório `runs/pesquisa_linear_01` precisa estar ausente antes de uma nova execução; use outro nome se já existir. "
              "O notebook escreve relatórios em `v19/evidence/`; não escreve `submission.csv`.", "",
              f'Notebook SHA-256: `{hashlib.sha256(notebook).hexdigest()}`.', "",
              "Versões e hashes dos dados estão no relatório JSON. Bibliotecas fixadas em `requirements.txt`; "
              "a lista do ambiente local está em `installed_versions_local.txt`.", ""]
    (DEST / "RESULTADOS_v19.md").write_text("\n".join(lines))
    rank_file = DEST / "rank.json"
    rank = json.loads(rank_file.read_text())
    rank["observacao"] = ("Pesquisa com folhas lineares descartada na triagem 0–1; partes 2–4 não executadas; nenhum CSV novo."
                           if report["status"] == "rejected_in_screening" else
                           "Pesquisa com folhas lineares descartada na confirmação; nenhum CSV novo.")
    rank_file.write_text(json.dumps(rank, ensure_ascii=False, indent=2) + "\n")
    archive = ROOT / "gci_home_credit_v19.zip"
    excluded = {"README.md", "RANK.md", "rank.json", "manifesto_sha256.json", archive.name}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for file in sorted(DEST.rglob("*")):
            if file.is_file() and file.name not in excluded:
                assert file.suffix != ".csv", "Negative research cannot contain a submission."
                z.write(file, str(file.relative_to(DEST)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    organize("v19")
    print(archive, archive.stat().st_size)


if __name__ == "__main__":
    package()
