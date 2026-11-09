"""Package real v24 results without inventing unexecuted folds or CSVs."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import pandas as pd
from organizar_entregaveis import organize

ROOT = Path(__file__).resolve().parent
DEST = ROOT.parent/"entregaveis/v24"


def copied(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert target.read_bytes() == source.read_bytes(), f"Preserve differing existing artifact: {target}"
    else:
        shutil.copy2(source, target)


def value(number):
    return "Não calculada" if number is None else f"{number:.8f}"


def table(rows, columns):
    lines = ["| "+" | ".join(title for key,title in columns)+" |",
             "|"+"|".join("---" for _ in columns)+"|"]
    for row in rows:
        cells = []
        for key,title in columns:
            item = row.get(key)
            cells.append("Não calculada" if item is None else f"{item:.8f}" if isinstance(item,float) else str(item))
        lines.append("| "+" | ".join(cells)+" |")
    return lines


def package():
    src = ROOT/"v24/evidence"
    r = json.loads((src/"experiment_report.json").read_text())
    assert r["status"] in {"rejected_in_screening","rejected_in_confirmation","gain_confirmed"}
    protocol = json.loads((src/"frozen_protocol.json").read_text())
    assert protocol == r["protocol"]
    screening_reference=json.loads((ROOT.parent/"entregaveis/v23/evidence/experiment_report.json").read_text())["screening_baseline_auc"]
    assert abs(r["screening_baseline_auc"]-screening_reference)<1e-12,"v21 screening reference was not reproduced."
    assert r["protocol_sha256"] == hashlib.sha256((src/"frozen_protocol.json").read_bytes()).hexdigest()
    assert all(row["threads"] == 2 and row["torch_interop_pool"] == 2 for row in r["fold_results"])
    assert all(pool["num_threads"] <= 2 for row in r["fold_results"] for pool in row["threadpools"])
    if r["status"] == "rejected_in_screening":
        assert r["confirmation_folds_executed"] == []
        assert sorted(row["fold"] for row in r["fold_results"]) == [0,1]
    elif "frozen_selection.json" in [p.name for p in src.iterdir()]:
        frozen = json.loads((src/"frozen_selection.json").read_text())
        assert frozen["status"] == "recipe_frozen_before_confirmation"
        assert "partes_2_4" not in {row["parte"] for row in frozen.get("table",[])}
        assert frozen["alpha"] == r["alpha"]
    positive = r["status"] == "gain_confirmed"
    proof = None
    if positive:
        proof = json.loads((src/"reproducibility.json").read_text())
        assert proof["identical_csv_bytes"] and all(t < 3600 for t in proof["seconds"])
    else:
        assert not list(DEST.rglob("submission*.csv"))
    for name in ["v24_fm.py","build_v24.py","repair_v24_preprocessing.py","check_v24.py","package_v24.py","build_v24_final.py","finalize_v24.py","execute_notebook.py","v17_definitions_sha256.json"]:
        copied(ROOT/name, DEST/name)
    copied(ROOT/"v24/gci_home_credit_v24_pesquisa.ipynb", DEST/"gci_home_credit_v24_pesquisa.ipynb")
    for path in src.iterdir():
        if path.is_file():
            copied(path, DEST/"evidence"/path.name)
    copied(ROOT/"v24/installed_versions_local.txt", DEST/"installed_versions_local.txt")
    if positive:
        copied(ROOT/"gci_home_credit_v24.ipynb",DEST/"gci_home_credit_v24.ipynb")
        copied(ROOT/"v24_final_1/runs/reproduction_1/submission.csv",DEST/"output/submission.csv")
        for run in [1,2]:
            directory=ROOT/f"v24_final_{run}/runs/reproduction_{run}"
            for name in ["source.ipynb","executed.ipynb","execution_report.json"]:
                copied(directory/name,DEST/f"evidence/reproduction_{run}"/name)
            copied(ROOT/f"v24_final_{run}/reproduction_{run}.log",DEST/f"evidence/reproduction_{run}/console.log")
    for attempt in ["pesquisa_01","pesquisa_02","pesquisa_03"]:
        run = ROOT/"v24/runs"/attempt
        if run.exists():
            for path in run.iterdir():
                if path.is_file():
                    copied(path, DEST/"evidence"/attempt/path.name)
            copied(ROOT/"v24"/(attempt+".log"), DEST/"evidence"/attempt/"console.log")
    failed = ROOT/"v24/attempt_01_sandbox_failure/evidence"
    for path in failed.iterdir():
        if path.is_file():
            copied(path, DEST/"evidence/attempt_01_sandbox_failure"/path.name)
    failed = ROOT/"v24/attempt_02_scaling_error"
    for path in failed.rglob("*"):
        if path.is_file():
            copied(path, DEST/"evidence/attempt_02_scaling_error"/path.relative_to(failed))
    for name in ["table","strata_table","rank_correlations"]:
        pd.DataFrame(r[name]).to_csv(DEST/(name+".csv"), index=False, float_format="%.17g")
    pd.DataFrame(r["screening_grid"]).to_csv(DEST/"screening_grid.csv",index=False,float_format="%.17g")
    current_rows = {row["parte"]:row for row in r["table"]}
    # Read historical metric records, never historical OOFs or predictions.
    original = json.loads((ROOT.parent/"entregaveis/v21/evidence/experiment_report.json").read_text())
    historical = {row["parte"]:row["v21"] for row in original["table"]}
    complete = []
    for label, baseline in historical.items():
        row = current_rows.get(label,{})
        complete.append(dict(parte=label,v21_historica_ranks_globais=baseline,
                             v21_recalculada=row.get("v21"),v24=row.get("v24"),fm=row.get("fm"),
                             delta=row.get("delta"),executada=bool(row)))
    pd.DataFrame(complete).to_csv(DEST/"auc_todas_as_partes.csv",index=False,float_format="%.17g")
    alpha = r.get("alpha",r["screening_choice"]["alpha"])
    weights = r.get("weights",r.get("evaluated_weights"))
    labels = ["LightGBM B + EXT","LightGBM B v7","MLP 8 sementes","CatBoost Plain + EXT",
              "TabM periódico","TabM por pares","CatBoost Ordered depth 5","FM PyTorch"]
    weight_rows = [dict(componente=label,peso=float(weight)) for label,weight in zip(labels,weights)]
    pd.DataFrame(weight_rows).to_csv(DEST/"pesos_avaliados.csv",index=False,float_format="%.17g")
    lines = ["# Resultado v24 — Factorization Machine PyTorch", "",
             "**Ganho confirmado e CSV reproduzido duas vezes.**" if positive else
             "**Sem ganho aprovado; nenhum CSV de submissão v24 foi gerado.**", "",
             "Pesquisa de 09/11/2026, encerramento até 14/11/2026. Uma única configuração FM foi executada, "
             "sem bibliotecas novas. Os sete componentes v21 foram retreinados a partir dos dois CSVs input/. "
             "Sem carregar previsões, OOFs, checkpoints ou modelos. Os arquivos NPZ gerados são evidências "
             "de saída, nunca entradas do notebook. Nenhuma variável SK_ID_CURR, transformação de ID, "
             "dado externo ou regra manual de previsão foi usado.", "",
             "## Protocolo", "",
             "Configuração e estratos constam em PLANO.md e evidence/frozen_protocol.json, congelados antes "
             "de qualquer treino. FM de segunda ordem, posto 16, termo linear, BCE sem pesos, AdamW "
             "lr=0,002/weight_decay=0,001, batch 2048, seed 42 e duas threads. Features plusx; "
             "categorias one-hot e numéricas contínuas por quantis normais. Todos os ajustes novos "
             "são feitos apenas no treino correspondente.", "",
             "Épocas escolhidas em validação interna de 15% do treino externo; depois novo modelo treinado "
             "do zero no treino externo completo. Inverter TARGET da parte externa, e também da validação "
             "interna quando aplicável, não alterou os frames supervisionados. O early stopping e o "
             "pré-processamento não supervisionado train+test da referência v21 foram preservados. "
             "Esse isolamento da FM não torna todo o pipeline herdado estritamente indutivo.", "",
             "Grade de peso [0%,5%,10%,15%,20%], preservando proporções v21. Empates pelo menor peso. "
             "Percentis ao quadrado; seleção somente com rótulos 0–1. Se a triagem passa, a mesma grade "
             "é reaplicada aos ranks globais com somente rótulos 0–1 antes de avaliar 2–4. "
             "Promoção exige delta 2–4 ≥0,00010 e delta total >0. Estratos são diagnóstico, sem regra "
             "adicional de promoção e sem uso em features ou pesos de treino.", "",
             "## Corte rápido 0–1", "",
             "Na triagem, os ranks usam somente as OOF disponíveis 0–1. Isso difere levemente dos ranks "
             "globais dos registros históricos da v21; a comparação nova usa os mesmos ranks disponíveis "
             "para referência e candidata.", ""]
    lines += table([dict(row,delta=row["auc"]-r["screening_baseline_auc"]) for row in r["screening_grid"]],
                   [("alpha","Peso FM"),("auc","AUC ensemble 0–1"),("delta","Delta v21")])
    lines += ["", f"Peso escolhido na triagem: {100*r['screening_choice']['alpha']:.0f}%.", ""]
    if r["status"] == "rejected_in_screening":
        lines += ["**Corte rápido acionado: 2–4 não foram treinadas nem avaliadas. Não houve treino integral, "
                  "CSV de submissão, SHA-256 de CSV ou duas reproduções.**", ""]
    lines += ["## AUCs por parte", ""]
    lines += table(complete,[("parte","Parte"),("v21_historica_ranks_globais","v21 histórica"),
                            ("v21_recalculada","v21 recalculada"),("v24","Candidata"),("fm","FM isolada"),("delta","Delta recalculado")])
    lines += ["", "Valores históricos são registros, não execuções novas. Células não calculadas respeitam o corte rápido. "
              "AUCs conjuntas são calculadas nas linhas reunidas, não por média das AUCs das partes.", "",
              "## Estratos de tarifa", "",
              "Chave round(AMT_CREDIT/AMT_ANNUITY,2); frequência apenas no treino externo. Raras=0–9 "
              "(inclui chaves inéditas), 10–49, 50–199 e ≥200, com ausentes/inválidas em separado. "
              "Cada percentual usa como denominador as linhas do respectivo conjunto avaliado. "
              "Definições e atribuições foram congeladas antes do primeiro fit em frozen_strata.csv. "
              "AUC não calculada indica ausência de linhas ou de ambas as classes.", ""]
    scope = "total" if "total" in current_rows else "partes_0_1"
    lines += table([row for row in r["strata_table"] if row["parte"] == scope],
                   [("estrato","Estrato"),("linhas","Linhas"),("percentual_linhas","% linhas"),
                    ("positivos","Positivos"),("v21","v21"),("v24","Candidata"),("delta","Delta"),("fm","FM isolada")])
    lines += ["", f"Escopo da tabela: {scope}. Tabelas por parte e demais agregados disponíveis em strata_table.csv.", "",
              "## Correlação de ranking com a FM", "",
              "Spearman das previsões OOF brutas de cada componente, antes dos percentis ao quadrado. "
              "Maior diversidade de ranking, sozinha, não implica ganho de AUC.", ""]
    lines += table([row for row in r["rank_correlations"] if row["parte"] == scope],
                   [("componente","Componente v21"),("spearman","Spearman com FM")])
    lines += ["", "## Pesos avaliados", ""]
    lines += table(weight_rows,[("componente","Componente"),("peso","Peso")])
    lines += ["", "Sem aprovação, esses pesos não constituem uma nova receita para submissão; "
              "a referência permanece v21. Nenhum ajuste foi feito após a triagem/confirmação.", "",
              "## O que foi executado de verdade", "",
              f"Pesquisa concluída em {r['seconds_including_prepare']/60:.2f} minutos, "
              f"{r['platform']}, Python {r['python'].split()[0]}. "
              "Dois workers de CV, duas threads por modelo; não é uma medição em apenas dois núcleos totais. "
              "Google Colab/Linux não foi executado. O runner incluído executa todas as células como Python, "
              "sem kernel Jupyter.", "",
              "A primeira tentativa foi interrompida pelo sandbox macOS (OpenMP/SHM/sysctl) antes dos treinos "
              "externos. O prepare havia ajustado os auxiliares EXT sem TARGET. Log e notebook com erro "
              "foram preservados.", "",
              "A segunda tentativa revelou um erro de implementação: colunas de contagem com dois níveis "
              "(NAME_CONTRACT_TYPE_COUNT e CODE_GENDER_COUNT, até 210.782) foram tratadas como indicadores "
              "e não normalizadas. AUCs FM dessa tentativa: 0,71212829 e 0,62620075; não são um teste válido "
              "da hipótese. A tentativa foi interrompida antes de selecionar peso ou executar 2–4. "
              "Código, protocolo, diagnóstico de escala e logs inválidos foram preservados.", "",
              "Correção única, registrada e congelada na revisão 2 antes da terceira tentativa: somente "
              "indicadores com valores em {0,1} pulam os quantis; contagens de dois níveis são normalizadas. "
              "Sem mudança de hiperparâmetros, seeds, arquitetura, estratos, grade ou promoção. "
              "A terceira tentativa começou em processo novo e retreinou a referência; todos os resultados "
              "apresentados nas tabelas pertencem a essa tentativa válida.", ""]
    if positive:
        lines += [f"Reproduções integrais: {proof['seconds'][0]/60:.2f} e {proof['seconds'][1]/60:.2f} minutos.",
                  f"SHA-256 1: `{proof['csv_sha256_run_1']}`.",
                  f"SHA-256 2: `{proof['csv_sha256_run_2']}`.", ""]
    else:
        lines += ["Não executados: treino final em 100%, dupla reprodução, verificação do limite final de "
                  "60 minutos e teste em Colab. Nenhum SHA-256 de CSV de submissão está disponível.", ""]
    lines += ["## Reprodução da pesquisa", "",
              "Coloque input/train.csv e input/test.csv ao lado do notebook de pesquisa e execute todas "
              "as células em runtime novo. Use pasta limpa para preservar evidências anteriores. "
              "Não é necessário nem permitido carregar os NPZ de evidência.", "", "```sh",
              "PYTHONHASHSEED=42 python execute_notebook.py --notebook gci_home_credit_v24_pesquisa.ipynb --name pesquisa_nova",
              "```", "", "Versões: "+"; ".join(f"{k} {v}" for k,v in r["exact_installed_versions"].items())+".", "",
              "Score público, posição e total de participantes v24 não informados; nenhuma submissão automática. "
              "Os folds já foram usados em pesquisas anteriores, portanto a confirmação não é um holdout intocado.", ""]
    (DEST/"RESULTADOS_v24.md").write_text("\n".join(lines))
    audit_path = DEST/"evidence/source_audit.json"
    audit = json.loads(audit_path.read_text())
    audit.update(final_status=r["status"], confirmation_folds_executed=r["confirmation_folds_executed"],
                 full_reproduction_not_executed=not positive,
                 completed_outer_folds=sorted(row["fold"] for row in r["fold_results"]),
                 target_flip_audits_passed=all(row["fm_details"]["target_flip_audit"] for row in r["fold_results"]))
    audit_path.write_text(json.dumps(audit,indent=2)+"\n")
    rank_path = DEST/"rank.json"
    rank = json.loads(rank_path.read_text())
    rank["observacao"] = ("FM PyTorch aprovada e CSV reproduzido duas vezes." if positive else
                          "FM PyTorch sem ganho aprovado; "+("corte rápido 0–1 acionado; 2–4 não executadas." if r["status"] == "rejected_in_screening" else "reprovada em 2–4/total."))
    rank_path.write_text(json.dumps(rank,ensure_ascii=False,indent=2)+"\n")
    organize("v24")
    archive = ROOT/"gci_home_credit_v24.zip"
    assert not archive.exists(), "Do not overwrite an existing version archive."
    excluded = {archive.name,"README.md","RANK.md","rank.json","manifesto_sha256.json"}
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(DEST.rglob("*")):
            if path.is_file() and path.name not in excluded:
                z.write(path,str(path.relative_to(DEST)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    organize("v24")
    print(archive, hashlib.sha256(archive.read_bytes()).hexdigest())


if __name__ == "__main__":
    package()
