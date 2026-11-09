"""Package v23 actual outcomes; no CSV before confirmed gain and two clean reruns."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from organizar_entregaveis import organize

ROOT = Path(__file__).resolve().parent
DEST = ROOT.parent/"entregaveis/v23"


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def package():
    directory = ROOT/"v23/evidence"
    r = json.loads((directory/"experiment_report.json").read_text())
    positive = r["confirmed"] and r["status"] == "gain_confirmed"
    assert positive or r["status"].startswith("rejected_")
    assert all(v["threads"] == 2 and v["torch_interop_pool"] == 2 for v in r["fold_results"])
    assert all(p["num_threads"] <= 2 for v in r["fold_results"] for p in v["threadpools"])
    if "table" in r:
        frozen = json.loads((directory/"frozen_selection.json").read_text())
        assert frozen["status"] == "recipe_frozen_before_confirmation" and "table" not in frozen
        assert frozen["alpha"] == r["alpha"] and frozen["weights"] == r["weights"]
    evidence = DEST/"evidence"
    copy(ROOT/"SLACK_AUDIT_20261030.md", evidence/"SLACK_AUDIT_20261030.md")
    copy(ROOT/"sources/official_README_20261020.md", evidence/"official_README_20261020.md")
    for name in ["v23_logodds.py", "build_v23.py", "build_v23_final.py", "finalize_v23.py", "package_v23.py", "execute_notebook.py", "v17_definitions_sha256.json"]:
        copy(ROOT/name, DEST/name)
    copy(ROOT/"v23/gci_home_credit_v23_pesquisa.ipynb", DEST/"gci_home_credit_v23_pesquisa.ipynb")
    for name in ["experiment_report.json", "frozen_selection.json"]:
        if (directory/name).exists():
            copy(directory/name, evidence/name)
    for name in ["source.ipynb", "executed.ipynb", "execution_report.json"]:
        copy(ROOT/"v23/runs/pesquisa_01"/name, evidence/"validation"/name)
    copy(ROOT/"v23/pesquisa_01.log", evidence/"validation/console.log")
    proof = None
    if positive:
        proof = json.loads((directory/"reproducibility.json").read_text())
        assert proof["identical_csv_bytes"]
        copy(directory/"reproducibility.json", evidence/"reproducibility.json")
        copy(ROOT/"gci_home_credit_v23.ipynb", DEST/"gci_home_credit_v23.ipynb")
        copy(ROOT/"v23_final_1/runs/reproduction_1/submission.csv", DEST/"output/submission.csv")
        for i in [1, 2]:
            for name in ["executed.ipynb", "execution_report.json"]:
                copy(ROOT/f"v23_final_{i}/runs/reproduction_{i}"/name, evidence/f"reproduction_{i}"/name)
            copy(ROOT/f"v23_final_{i}/reproduction_{i}.log", evidence/f"reproduction_{i}/console.log")
    else:
        assert not list(DEST.rglob("submission*.csv"))
    audit_path = evidence/"source_audit.json"
    audit = json.loads(audit_path.read_text())
    if not (evidence/"research_source_audit.json").exists():
        copy(audit_path, evidence/"research_source_audit.json")
    audit["full_reproduction_not_executed"] = not positive
    audit["final_status"] = r["status"]
    audit["confirmation_folds_executed"] = r["confirmation_folds_executed"]
    if positive:
        audit["final_notebook_sha256"] = proof["notebook_sha256"]
        audit["csv_identical_in_two_fresh_processes"] = True
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n")
    versions = r["exact_installed_versions"]
    (DEST/"requirements.txt").write_text("".join(f"{k}=={v}\n" for k, v in versions.items()))
    if (ROOT/"v23/installed_versions_local.txt").exists():
        copy(ROOT/"v23/installed_versions_local.txt", DEST/"installed_versions_local.txt")
    lines = ["# Resultado v23 — ensemble em log-odds", "",
             "**Ganho confirmado; CSV reproduzido duas vezes.**" if positive else "**Sem ganho aprovado; nenhum CSV v23 foi gerado.**", "",
             "Referência: v21, público informado0,77876 e posição38 em28/10/2026, total de participantes não informado. "
             "Melhor público registrado: v20,0,77877,38/2.741 em26/10/2026. Variação pública de0,00001 não demonstra superioridade privada.", "",
             "Todos os sete componentes da v21 treinados novamente a partir dos dois arquivos input/train.csv e input/test.csv. "
             "Sem carregar previsões, OOFs, modelos ou checkpoints. Sem feature de ID, dados externos ou regras manuais de previsão. "
             "As 24 definições originais da v17 foram preservadas. Supervised.add exclui os rótulos da parte avaliada. "
             "O pré-processamento não supervisionado herdado da v17 usa covariáveis train+test; o README examinado não esclarece expressamente essa opção.", "",
             "Única hipótese: média ponderada de log-odds com pesos dos sete componentes fixos. Clip1e-6 apenas para domínio numérico do log. "
             "Transformar o score agregado em percentil ao quadrado e combinar com o blend atual. "
             "Alpha na grade0,0.1,...,1, escolhido somente em0–1; empates preferem menor alpha. Alpha1 equivale à ordenação da média em log-odds. "
             "Não houve treino de meta-modelo ou recalibração. Probabilidades dos componentes não são necessariamente calibradas.", "",
             "StratifiedKFold(5,shuffle=True,random_state=42), ordem original. Na triagem, percentis somente nas OOFs disponíveis0–1. "
             "Se a triagem passar, completar OOFs e refixar alpha "
             "apenas com rótulos0–1 e percentis globais. Salvar frozen_selection.json antes de medir2–4 e total. Nenhum ajuste após confirmação.", "",
             "## Seleção nas partes0–1", "",
             f'AUC de referência na triagem: {r["screening_baseline_auc"]:.8f}.', "",
             "| Alpha log-odds | AUC0–1 | Diferença |", "|---|---:|---:|"]
    for row in r["screening_grid"]:
        lines.append(f'| {row["alpha"]:.1f} | {row["auc"]:.8f} | {row["auc"]-r["screening_baseline_auc"]:+.8f} |')
    if "table" in r:
        lines += ["", "## Confirmação", "", "| Parte | v21 | v23 | Diferença |", "|---|---:|---:|---:|"]
        for row in r["table"]:
            lines.append(f'| {row["parte"]} | {row["v21"]:.8f} | {row["v23"]:.8f} | {row["delta"]:+.8f} |')
        lines += ["", f'Alpha congelado: {r["alpha"]:.1f}.', "",
                  "| Componente | Peso dentro de cada blend | Árvores/épocas |", "|---|---:|---:|"]
        labels = ["LightGBM B + EXT auxiliares", "LightGBM B — variáveis v7", "MLP — oito sementes", "CatBoost Plain", "TabM periódico", "TabM periódico por pares", "CatBoost Ordered depth5"]
        for label, weight, size in zip(labels, r["weights"], r["final_sizes"]):
            lines.append(f"| {label} | {weight:.9f} | {size} |")
        lines += ["", "Esses pesos são usados em ambos os blends. Score final=(1−alpha)×blend atual+alpha×percentil² da média ponderada dos log-odds. "
                  "Tamanhos calculados para treino integral=média do early stopping×1,15, arredondada. "
                  "Quando a candidata é reprovada, esses tamanhos não são usados e os pesos avaliados não constituem uma receita aprovada.", ""]
    else:
        lines += ["", "Triagem reprovada: partes2–4 e AUC total não executadas. Nenhum novo peso aprovado.", ""]
    lines += ["## Execução real e reprodução", "",
              f'Validação: {r["seconds_including_prepare"]/60:.2f} minutos; {r["platform"]}, Python {r["python"].split()[0]}. '
              "Dois processos de CV independentes, duas threads por modelo. Colab/Linux não executado.", ""]
    if positive:
        lines += [f'Duas reproduções integrais: {proof["seconds"][0]/60:.2f} e {proof["seconds"][1]/60:.2f} minutos, '
                  "em processos Python novos e pastas separadas. Todas as células executadas pelo runner incluído, sem kernel Jupyter.", "",
                  f'SHA-256 CSV, execução1: `{proof["csv_sha256_run_1"]}`.',
                  f'SHA-256 CSV, execução2: `{proof["csv_sha256_run_2"]}`.', "",
                  "CSV com61.500 linhas e colunas SK_ID_CURR,TARGET, IDs na ordem do teste. Público e rank v23 não informados; não houve submissão automática.", "",
                  "Para reproduzir, coloque input/ ao lado do notebook final e execute todas as células em runtime novo. "
                  "A execução padrão treina tudo do zero e grava output/submission.csv.", "", "```sh",
                  "GCI_RUN_CV=0 GCI_CV_ONLY=0 PYTHONHASHSEED=42 python execute_notebook.py --notebook gci_home_credit_v23.ipynb --name reproducao_nova", "```", ""]
    else:
        lines += ["Nenhum treino integral novo, CSV, SHA-256 de CSV ou dupla reprodução foi executado após a reprovação. "
                  "Para repetir a pesquisa, coloque input/ ao lado do notebook de pesquisa e execute todas as células.", "", "```sh",
                  "PYTHONHASHSEED=42 python execute_notebook.py --notebook gci_home_credit_v23_pesquisa.ipynb --name pesquisa_nova", "```", ""]
    lines += ["Use um nome novo por execução. O ZIP inclui código e evidências; não inclui os dados.", "",
              "Versões: "+"; ".join(f"{k} {v}" for k, v in versions.items())+".", "",
              "Hipótese trazida pelo resumo do Slack do usuário: https://gciworld2026sept3.slack.com/archives/C0BDW0TT2HZ/p1792965992208839 . "
              "Nenhuma thread foi consultada ao vivo. Ganhos relatados por outros participantes não garantem ganho aqui. "
              "Os folds foram reutilizados em pesquisas anteriores; a confirmação não é uma estimativa independente de todas as tentativas.", ""]
    (DEST/"RESULTADOS_v23.md").write_text("\n".join(lines))
    rank_path = DEST/"rank.json"
    rank = json.loads(rank_path.read_text())
    rank["observacao"] = "Ganho local confirmado e CSV reproduzido duas vezes; score público e rank não informados." if positive else "Blend em log-odds sem ganho aprovado; nenhum CSV v23."
    rank["fonte"] = "Pesquisa local de30/10/2026; sem submissão v23"
    if positive:
        rank["csv_sha256"] = proof["csv_sha256_run_1"]
    rank_path.write_text(json.dumps(rank, ensure_ascii=False, indent=2)+"\n")
    archive = ROOT/"gci_home_credit_v23.zip"
    excluded = {archive.name, "README.md", "RANK.md", "rank.json", "manifesto_sha256.json"}
    assert not archive.exists(), "Preserve previous version archives."
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(DEST.rglob("*")):
            if path.is_file() and path.name not in excluded:
                z.write(path, str(path.relative_to(DEST)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    organize("v23")
    print(archive, hashlib.sha256(archive.read_bytes()).hexdigest())


if __name__ == "__main__":
    package()
