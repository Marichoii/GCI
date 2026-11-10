"""Inspect official wheel metadata; never install packages or train models."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"v21_colab/evidence"
OUT.mkdir(parents=True,exist_ok=True)
requirements={line.split("==")[0]:line.split("==")[1] for line in (ROOT.parent/"entregaveis/v21/requirements.txt").read_text().splitlines() if line.strip()}


def fetch(url):
    with urlopen(Request(url,headers={"User-Agent":"GCI-v21-Colab-compatibility-audit"}),timeout=60) as response:
        return response.read()


def package(item):
    name,version=item
    url=f"https://pypi.org/pypi/{name}/{version}/json"
    data=json.loads(fetch(url))
    wheels=[dict(filename=f["filename"],url=f["url"],sha256=f["digests"]["sha256"],
                 requires_python=f["requires_python"],yanked=f.get("yanked",False))
            for f in data["urls"] if f["packagetype"]=="bdist_wheel"]
    return name,dict(version=version,requires_python=data["info"]["requires_python"],
                     requires_dist=data["info"]["requires_dist"],source=url,wheels=wheels)


class Links(HTMLParser):
    def __init__(self):
        super().__init__();self.urls=[]
    def handle_starttag(self,tag,attrs):
        if tag=="a":
            self.urls += [v for k,v in attrs if k=="href"]


packages=dict(ThreadPoolExecutor(max_workers=6).map(package,requirements.items()))
torch_index="https://download.pytorch.org/whl/cpu/torch/"
html=fetch(torch_index).decode()
parser=Links();parser.feed(html)
cpu_urls=[url for url in parser.urls if "torch-2.6.0+cpu-" in unquote(url) and "x86_64.whl" in unquote(url)]
targets={}
for py in ["311","312","313","314"]:
    rows=[]
    for name,record in packages.items():
        if name=="torch":
            matches=[unquote(u).split("/")[-1].split("#")[0] for u in cpu_urls
                     if f"-cp{py}-cp{py}-" in unquote(u) and ("manylinux" in u or "linux_x86_64" in u)]
        else:
            matches=[]
            for wheel in record["wheels"]:
                filename=wheel["filename"]
                pure=filename.endswith("-none-any.whl") and ("-py3-" in filename or "-py2.py3-" in filename)
                abi_independent="-py3-none-" in filename and "manylinux" in filename and "x86_64" in filename
                native=f"-cp{py}-cp{py}-" in filename and "manylinux" in filename and "x86_64" in filename
                if not wheel["yanked"] and (pure or abi_independent or native):matches.append(filename)
        rows.append(dict(package=name,version=record["version"],available=bool(matches),matching_wheels=matches))
    targets["cp"+py]=rows
report=dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
    purpose="Wheel metadata only; not installation, Colab execution or training",packages=packages,
    torch_cpu_index=torch_index,torch_cpu_wheel_urls=cpu_urls,targets=targets)
(OUT/"wheel_metadata.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
for target,rows in targets.items():
    print(target,[(r["package"],r["available"]) for r in rows])
