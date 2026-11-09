"""Copy versioned deliverables without changing trained notebooks or CSV bytes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

COMPETITION = Path(__file__).resolve().parent
DELIVERABLES = COMPETITION.parent / "entregaveis"


def update_index():
    records = [json.loads(p.read_text()) for p in DELIVERABLES.glob("v*/rank.json")]
    records.sort(key=lambda r: int(r["versao"][1:]))
    columns = ["versao", "score_publico", "rank_publico", "participantes", "data_relato", "fonte", "notion_url"]
    with (DELIVERABLES / "ranking.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)
    lines = ["# Entregáveis — Home Credit", "", "Somente a competição Home Credit. O trabalho final de Telecom / Company A fica fora do escopo deste repositório.", "",
             "Uma pasta por versão. Score e rank são relatos históricos, não consultas ao placar ao vivo.", "",
             "| Versão | Score público | Rank informado | Data do relato | Entrega |",
             "|---|---:|---:|---|---|"]
    notion_file = DELIVERABLES / "notion.json"
    if notion_file.exists():
        notion = json.loads(notion_file.read_text())
        location = "registro privado" if notion.get("private_draft") else "na página da competição"
        lines[2:2] = [f'[Abrir entregáveis e ranking no Notion]({notion["page_url"]}) — {location}.', ""]
    for r in records:
        score = f'{r["score_publico"]:.5f}' if r.get("score_publico") is not None else "Não informado"
        rank = str(r["rank_publico"]) if r.get("rank_publico") is not None else "Não informado"
        if r.get("participantes"):
            rank += f' / {r["participantes"]}'
        v = r["versao"]
        lines.append(f'| [{v}]({v}/README.md) | {score} | {rank} | {r.get("data_relato") or "Não informada"} | {r.get("situacao", "")} |')
    lines += ["", "Cada pasta contém `rank.json`, `RANK.md` e um manifesto SHA-256 dos artefatos locais.",
              "As observações anteriores ficam no histórico da versão; arquivos ausentes não são recriados ou apresentados como disponíveis.", ""]
    (DELIVERABLES / "README.md").write_text("\n".join(lines))


def organize(version, *, score=None, rank=None, participants=None, reported_on=None, note=None):
    assert version.startswith("v") and version[1:].isdigit()
    target = DELIVERABLES / version
    target.mkdir(parents=True, exist_ok=True)
    archive = COMPETITION / f"gci_home_credit_{version}.zip"
    notebook_name = f"gci_home_credit_{version}.ipynb"
    if archive.is_file():
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            for member in z.infolist():
                path = PurePosixPath(member.filename)
                assert not path.is_absolute() and ".." not in path.parts
                assert (member.external_attr >> 16) & 0o170000 != 0o120000
                if member.is_dir():
                    continue
                dest = target.joinpath(*path.parts)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(z.read(member))
        shutil.copy2(archive, target / archive.name)
    else:
        for source in [COMPETITION / notebook_name, COMPETITION / "sources" / notebook_name]:
            if source.is_file():
                shutil.copy2(source, target / notebook_name)
                break
    path = target / "rank.json"
    record = json.loads(path.read_text()) if path.exists() else {"versao": version, "historico": []}
    for key, value in [("score_publico", score), ("rank_publico", rank), ("participantes", participants), ("data_relato", reported_on)]:
        if value is not None:
            record[key] = value
        else:
            record.setdefault(key, None)
    if note is not None:
        record["observacao"] = note
    record.setdefault("fonte", "Informado pelo usuário nesta conversa")
    record.setdefault("notion_url", None)
    has_csv = any(target.glob("**/submission*.csv"))
    has_notebook = any(target.glob("*.ipynb"))
    record["situacao"] = ("Pacote com evidências de reprodução" if (target / "evidence/reproducibility.json").exists()
                          else "Arquivos históricos" if has_notebook and has_csv
                          else "Notebook histórico; CSV não disponível localmente" if has_notebook
                          else "Registro histórico; arquivos ainda não disponíveis localmente")
    experiment_file = target / "evidence/experiment_report.json"
    if experiment_file.exists() and not has_csv:
        experiment = json.loads(experiment_file.read_text())
        status = experiment.get("status", "running")
        record["situacao"] = ("Pesquisa sem ganho; sem CSV" if status.startswith("rejected_")
                              else "Ganho em validação; reprodução pendente" if status == "gain_confirmed"
                              else "Pesquisa em execução; sem CSV")
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    rank_text = str(record["rank_publico"]) if record["rank_publico"] is not None else "Não informado"
    if record["participantes"]:
        rank_text += f' / {record["participantes"]}'
    score_text = f'{record["score_publico"]:.5f}' if record["score_publico"] is not None else "Não informado"
    rank_lines = [f'# {version} — score e rank', "", f'- Score público: {score_text}', f'- Rank informado: {rank_text}',
                  f'- Data do relato: {record["data_relato"] or "Não informada"}', f'- Fonte: {record["fonte"]}',
                  "", record.get("observacao", ""), "", "Posição histórica; não representa uma consulta ao ranking ao vivo.", ""]
    if record.get("historico"):
        rank_lines += ["## Histórico", ""]
        for entry in record["historico"]:
            rank_lines.append("- " + entry["descricao"])
        rank_lines.append("")
    (target / "RANK.md").write_text("\n".join(rank_lines))
    artifact_paths = sorted(p for p in target.rglob("*") if p.is_file() and p.name not in {"manifesto_sha256.json", "README.md", "rank.json", "RANK.md", "notion.json"})
    manifest = {str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifact_paths}
    (target / "manifesto_sha256.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    lines = [f'# Entregáveis {version}', "", record["situacao"], "", f'Score público: **{score_text}**. Rank informado: **{rank_text}**.',
             "", record.get("observacao", ""), "", "## Arquivos", ""]
    if record.get("notion_url"):
        lines[2:2] = [f'[Abrir esta entrega no Notion]({record["notion_url"]})', ""]
    lines.extend(f'- [{name}]({name})' for name in manifest)
    lines += ["", "[Score, rank e histórico](RANK.md) · [Manifesto SHA-256](manifesto_sha256.json)", ""]
    (target / "README.md").write_text("\n".join(lines))
    update_index()
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    parser.add_argument("--score", type=float)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--total", type=int)
    parser.add_argument("--data")
    parser.add_argument("--nota")
    a = parser.parse_args()
    print(organize(a.version, score=a.score, rank=a.rank, participants=a.total, reported_on=a.data, note=a.nota))
