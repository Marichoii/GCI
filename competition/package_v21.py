"""Package actual v21 research outcomes; CSV requires proof of two reruns."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from organizar_entregaveis import organize

ROOT=Path(__file__).resolve().parent
DEST=ROOT.parent/"entregaveis/v21"


def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)


def package():
    evidence_root=ROOT/"v21/v21/evidence"
    r=json.loads((evidence_root/"experiment_report.json").read_text())
    positive=r["confirmed"] and r["status"]=="gain_confirmed"
    assert positive or r["status"].startswith("rejected_")
    assert all(v["threads"]==2 and v["torch_interop_pool"]==2 for v in r["fold_results"])
    assert all(p["num_threads"]<=2 for v in r["fold_results"] for p in v["threadpools"])
    if "table" in r:
        frozen=json.loads((evidence_root/"frozen_selection.json").read_text())
        assert frozen["status"]=="recipe_frozen_before_confirmation" and "table" not in frozen
        assert frozen["weights"]==r["weights"] and frozen["alpha"]==r["alpha"]
        assert frozen["selected_candidate"]==r["selected_candidate"]
    evidence=DEST/"evidence"
    evidence.mkdir(exist_ok=True,parents=True)
    for name in ["v21_search.py","execute_notebook.py","v17_definitions_sha256.json"]:
        copy(ROOT/name,DEST/name)
    copy(ROOT/"v21/gci_home_credit_v21_pesquisa.ipynb",DEST/"gci_home_credit_v21_pesquisa.ipynb")
    for name in ["experiment_report.json","frozen_selection.json"]:
        if (evidence_root/name).exists():copy(evidence_root/name,evidence/name)
    directory=ROOT/"v21/runs/pesquisa_01"
    for name in ["source.ipynb","executed.ipynb","execution_report.json"]:
        copy(directory/name,evidence/"validation"/name)
    copy(ROOT/"v21/pesquisa_01.log",evidence/"validation/console.log")
    proof=None
    if positive:
        proof=json.loads((evidence_root/"reproducibility.json").read_text())
        assert proof["identical_csv_bytes"]
        copy(evidence_root/"reproducibility.json",evidence/"reproducibility.json")
        copy(ROOT/"gci_home_credit_v21.ipynb",DEST/"gci_home_credit_v21.ipynb")
        copy(ROOT/"v21_final_1/runs/reproduction_1/submission.csv",DEST/"output/submission.csv")
        for i in [1,2]:
            directory=ROOT/f"v21_final_{i}/runs/reproduction_{i}"
            for name in ["executed.ipynb","execution_report.json"]:
                copy(directory/name,evidence/f"reproduction_{i}"/name)
            copy(ROOT/f"v21_final_{i}/reproduction_{i}.log",evidence/f"reproduction_{i}/console.log")
    else:
        assert not list(DEST.rglob("submission*.csv"))
    versions=r["exact_installed_versions"]
    (DEST/"requirements.txt").write_text("".join(f"{k}=={v}\n" for k,v in versions.items()))
    if (ROOT/"v21/installed_versions_local.txt").exists():
        copy(ROOT/"v21/installed_versions_local.txt",DEST/"installed_versions_local.txt")
    lines=["# Home Credit — resultado v21", "",
           "**Ganho confirmado; CSV reproduzido duas vezes.**" if positive else
           "**Não houve ganho aprovado; nenhum CSV v21 foi gerado.**", "",
           "Referência: v20 (público informado 0,77877, rank histórico 38/2.741 em 26/10/2026). "
           "Somente input/train.csv e input/test.csv; todos os seis componentes da referência treinados novamente. "
           "Nenhuma previsão, OOF, checkpoint ou modelo salvo carregado. Nenhuma feature de ID, nenhum dado externo e nenhuma regra manual de previsão. "
           "24 definições originais da v17 preservadas; Supervised fora da parte e verificação por inversão do TARGET avaliado.", "",
           "Candidatos fixados antes da execução: CatBoost Ordered depth 5 e 6, lr 0,04, L2=5, seed 42; "
           "LightGBM B com extra_trees=True e extra_seed=42. São esquemas de treinamento diferentes dos CatBoost Plain descartados anteriormente.", "",
           "StratifiedKFold(5, shuffle=True, random_state=42), ordem original. Escolher uma família e peso novo "
           "na grade 0–0,30 com passo 0,025 apenas nas partes 0–1. Os seis pesos antigos mantêm suas proporções; percentis ao quadrado. "
           "Empates preferem peso menor. Sem ganho na triagem, não avaliar 2–4. "
           "Se passar, completar OOFs e refixar o peso com percentis globais usando somente rótulos 0–1; "
           "gravar frozen_selection.json antes de medir 2–4 e total. Não ajustar após confirmação.", "",
           "## Seleção nas partes 0–1", "",
           f'AUC de referência com percentis das linhas disponíveis: {r["screening_baseline_auc"]:.8f}.', "",
           "| Candidato | Peso novo | Melhor AUC do ensemble | Diferença |", "|---|---:|---:|---:|"]
    for row in r["screening"]:
        label=r["configurations"][row["candidate"]]["label"]
        lines.append(f'| {label} | {row["alpha"]:.3f} | {row["auc"]:.8f} | {row["auc"]-r["screening_baseline_auc"]:+.8f} |')
    if "table" in r:
        lines += ["", "## Confirmação com percentis globais", "",
                  "| Parte | v20 | v21 | Diferença |", "|---|---:|---:|---:|"]
        for row in r["table"]:
            lines.append(f'| {row["parte"]} | {row["v20"]:.8f} | {row["v21"]:.8f} | {row["delta"]:+.8f} |')
        lines += ["", f'Componente selecionado: `{r["selected_candidate"]}`. Peso novo: {r["alpha"]:.3f}.',
                  "", "| Componente | Peso final | Árvores / épocas calculadas |", "|---|---:|---:|"]
        labels=["LightGBM B + EXT auxiliares", "LightGBM B — variáveis v7", "MLP — 8 sementes", "CatBoost original",
                "TabM periódico original", "TabM periódico por pares", r["selected_candidate"]["label"]]
        for label,weight,size in zip(labels,r["weights"],r["baseline_sizes"]+[r["candidate_size"]]):
            lines.append(f"| {label} | {weight*100:.7f}% | {size} |")
        lines += ["",
                  f'Tamanhos calculados para treino integral: referência `{r["baseline_sizes"]}`; novo `{r["candidate_size"]}`. '
                  'Média do early stopping × 1,15, arredondada; a execução integral depende da aprovação e está documentada abaixo.', ""]
    else:
        lines += ["", "Interrompida na triagem: partes 2–4 não executadas nesta pesquisa. "
                  "Não há AUC v21 de confirmação ou total.", ""]
    lines += ["## Execução real", "",
              f'Pesquisa: {r["total_seconds_including_prepare"]/60:.2f} minutos de tempo decorrido, '
              f'{r["platform"]}, Python {r["python"].split()[0]}. CV com dois processos independentes e duas threads por modelo. '
              "Tempo e bytes no Colab/Linux não foram verificados.", ""]
    if positive:
        lines += [f'Duas reproduções integrais: {proof["seconds"][0]/60:.2f} e {proof["seconds"][1]/60:.2f} minutos. '
                  "Todas as células de código executadas pelo runner Python incluído, sem kernel Jupyter, em dois processos novos "
                  "e diretórios separados. As duas reproduções foram iniciadas em paralelo; cada uma treina seus componentes sequencialmente.", "",
                  f'CSV SHA-256, execução 1: `{proof["csv_sha256_run_1"]}`.',
                  f'CSV SHA-256, execução 2: `{proof["csv_sha256_run_2"]}`.', "",
                  "CSV com 61.500 linhas, colunas SK_ID_CURR,TARGET e IDs na ordem de input/test.csv. "
                  "Score público e rank da v21 aguardam submissão; melhora local não garante ganho no placar.", "",
                  "Para reproduzir: coloque gci_home_credit_v21.ipynb ao lado da pasta input e execute todas as células em runtime novo. "
                  "O padrão treina tudo e grava output/submission.csv, sem depender do CSV incluído. "
                  "AUCs da validação são registros fixos; GCI_RUN_CV=1 refaz a pesquisa. GCI_CV_ONLY=1 evita treino integral nessa opção.", ""]
        lines += ["Alternativa sem Jupyter, igual ao método realmente executado nesta verificação, em processo Python novo:", "",
                  "```sh", "GCI_RUN_CV=0 GCI_CV_ONLY=0 PYTHONHASHSEED=42 python execute_notebook.py --notebook gci_home_credit_v21.ipynb --name reproducao_nova", "```", "",
                  "Use outro nome para cada execução; o runner preserva os arquivos da execução anterior.", ""]
    else:
        lines += ["Nenhum treino integral de candidata, CSV ou dupla reprodução foi executado após a reprovação. "
                  "A v20 permanece o melhor CSV validado e enviado conhecido.", "",
                  "Para repetir a pesquisa: coloque o notebook de pesquisa ao lado de input e execute todas as células, "
                  "ou use execute_notebook.py com --notebook gci_home_credit_v21_pesquisa.ipynb e --name com um novo nome de execução.", ""]
    lines += ["## Versões e fontes", "", "Python "+r["python"].split()[0]+"; "+"; ".join(f"{k} {v}" for k,v in versions.items())+".", "",
              "[CatBoost Ordered](https://catboost.ai/docs/en/references/training-parameters/common#boosting_type) · "
              "[LightGBM extra_trees](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#extra_trees).", "",
              "Os folds já foram usados em pesquisas anteriores. Esta confirmação é um critério operacional, "
              "não uma estimativa independente de todas as tentativas. O top 10 fornecido pelo usuário não revela os métodos usados nem estabelece um teto teórico para modelos permitidos.", ""]
    (DEST/"RESULTADOS_v21.md").write_text("\n".join(lines))
    p=DEST/"rank.json"
    rank=json.loads(p.read_text())
    rank["observacao"]="Ganho local confirmado e CSV reproduzido duas vezes; público e rank pendentes." if positive else "Pesquisa sem ganho aprovado; nenhum CSV v21. V20 preservada como referência."
    rank["fonte"]="Pesquisa local de 28/10/2026; sem submissão v21"
    p.write_text(json.dumps(rank,ensure_ascii=False,indent=2)+"\n")
    archive=ROOT/"gci_home_credit_v21.zip"
    exclude={archive.name,"README.md","RANK.md","rank.json","manifesto_sha256.json"}
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(DEST.rglob("*")):
            if f.is_file() and f.name not in exclude:z.write(f,str(f.relative_to(DEST)))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    organize("v21")
    print(archive,archive.stat().st_size,hashlib.sha256(archive.read_bytes()).hexdigest())


if __name__=="__main__":package()
