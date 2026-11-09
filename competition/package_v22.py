"""Archive the completed, rejected density feature probe without inventing a CSV."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from organizar_entregaveis import organize

ROOT = Path(__file__).resolve().parent
DEST = ROOT.parent / "entregaveis/v22"


def package():
    report = json.loads((ROOT / "v22/evidence/probe_report.json").read_text())
    assert report["status"] == "rejected_in_feature_probe"
    assert not report["confirmed"] and report["csv_sha256"] is None
    assert report["chosen"] is None and report["confirmation_folds_executed"] == []
    assert not list(DEST.rglob("submission*.csv"))
    for fold in report["folds"]:
        assert fold["threads"] == 2
        assert all(p["num_threads"] <= 2 for p in fold["threadpools"])
    evidence = DEST / "evidence"
    validation = evidence / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "v22/evidence/probe_report.json", evidence / "probe_report.json")
    shutil.copy2(ROOT / "v22/evidence/probe_report.json", evidence / "experiment_report.json")
    for name in ["source.ipynb", "executed.ipynb", "execution_report.json"]:
        shutil.copy2(ROOT / "v22/runs/probe_01" / name, validation / name)
    shutil.copy2(ROOT / "v22/probe_01.log", validation / "console.log")
    shutil.copy2(ROOT / "v22/gci_home_credit_v22_pesquisa.ipynb", DEST / "gci_home_credit_v22_pesquisa.ipynb")
    for name in ["v22_density.py", "build_v22_probe.py", "execute_notebook.py", "v17_definitions_sha256.json"]:
        shutil.copy2(ROOT / name, DEST / name)
    versions = report["versions"]
    (DEST / "requirements.txt").write_text("".join(f"{k}=={v}\n" for k, v in versions.items()))
    lines = ["# Resultado v22 — densidade do empréstimo", "",
             "**Sem ganho aprovado na triagem; nenhum CSV v22 foi gerado.**", "",
             "Execução concluída em 28/10/2026; fechamento dos entregáveis em 30/10/2026. "
             "A comparação é uma ablação de LightGBM B com uma semente (42), não uma comparação entre ensembles completos. "
             "O filtro previamente declarado exigia AUC conjunta 0–1 e média das diferenças por parte positivas. Ambas as variantes falharam.", "",
             "| Parte | LightGBM de referência | Densidade geral | Densidade por contrato |",
             "|---|---:|---:|---:|"]
    for fold in report["folds"]:
        aucs = [d["auc"] for d in fold["details"]]
        lines.append(f'| {fold["fold"]} | {aucs[0]:.8f} | {aucs[1]:.8f} | {aucs[2]:.8f} |')
    lines.append(f'| 0–1 juntas | {report["probe_baseline_auc"]:.8f} | {report["candidates"][0]["auc_0_1"]:.8f} | {report["candidates"][1]["auc_0_1"]:.8f} |')
    lines += ["| 2–4 | Não executado | Não executado | Não executado |",
              "| Total | Não executado | Não executado | Não executado |", "",
              "Não há pesos novos, treino integral, SHA-256 de CSV ou dupla reprodução de CSV nesta versão. "
              "A reprovação desta implementação não prova que toda forma de KDE ou seu uso em um ensemble falharia.", "",
              "StratifiedKFold(5, shuffle=True, random_state=42), ordem original do arquivo. "
              "Ajuste da densidade somente nas linhas de treino da parte, usando AMT_CREDIT, AMT_ANNUITY e NAME_CONTRACT_TYPE. "
              "Sem TARGET ou SK_ID_CURR no estimador. Supervised.add exclui o TARGET da parte avaliada; "
              "a inversão dos rótulos de validação não alterou os frames supervisionados. As 24 definições originais da v17 foram preservadas.", "",
              "O pré-processamento original continua usando covariáveis de train+test em etapas não supervisionadas. "
              "Portanto, somente a nova densidade foi ajustada estritamente no treino de cada parte; "
              "esta execução não comprova um pipeline totalmente indutivo.", "",
              f'Tempo real: {report["seconds_including_prepare"]/60:.2f} minutos incluindo prepare; '
              f'{report["platform"]}, Python {report["python"].split()[0]}. '
              "Dois processos de validação, duas threads por modelo. Colab/Linux não executado.", "",
              "O notebook de pesquisa inclui todo o código e refaz a triagem a partir de input/train.csv e input/test.csv. "
              "Não carrega previsões, OOFs, modelos ou checkpoints. Coloque input/ ao lado do notebook e execute todas as células. "
              "O ZIP não inclui os dados.", "", "```sh",
              "PYTHONHASHSEED=42 python execute_notebook.py --notebook gci_home_credit_v22_pesquisa.ipynb --name pesquisa_nova",
              "```", "", "Use um nome novo por execução. O runner executa as células em um processo Python; não usa kernel Jupyter.", "",
              "Versões: " + "; ".join(f"{k} {v}" for k, v in versions.items()) + ".", "",
              "Ideia fornecida no resumo do Slack pelo usuário, atribuída a Theophile Avenel e sameenakb. "
              "Implementação própria com histograma suavizado, largura de banda Silverman, detecção de picos e mínimos. "
              "Os ganhos de outros participantes não foram reproduzidos aqui.", ""]
    (DEST / "RESULTADOS_v22.md").write_text("\n".join(lines))
    rank_path = DEST / "rank.json"
    rank = json.loads(rank_path.read_text())
    rank["fonte"] = "Pesquisa local executada em 28/10/2026; fechamento em 30/10/2026; sem submissão v22"
    rank["observacao"] = "Densidade geral e por contrato reprovadas na ablação LightGBM das partes 0–1; partes 2–4 e total não executadas. Nenhum CSV v22."
    rank_path.write_text(json.dumps(rank, ensure_ascii=False, indent=2) + "\n")
    archive = ROOT / "gci_home_credit_v22.zip"
    excluded = {archive.name, "README.md", "RANK.md", "rank.json", "manifesto_sha256.json"}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(DEST.rglob("*")):
            if path.is_file() and path.name not in excluded:
                z.write(path, str(path.relative_to(DEST)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert not any(n.endswith(".csv") for n in z.namelist())
    organize("v22")
    print(archive, hashlib.sha256(archive.read_bytes()).hexdigest())


if __name__ == "__main__":
    package()
