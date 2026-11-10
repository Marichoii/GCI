# Ambiente Colab — não altera modelos, seeds, pesos ou FROZEN_V21.
import os, sys, subprocess, importlib.metadata, random, platform
from pathlib import Path
import time as _colab_time
import json as _colab_json
import sysconfig as _colab_sysconfig

notebook_wall_started = _colab_time.time()
_COLAB_RUN_STARTED = _colab_time.perf_counter()
_COLAB_ENV_STARTED = _COLAB_RUN_STARTED
_V21_ENV_READY = False
if globals().get("_V21_RESTART_REQUIRED", False):
    raise RuntimeError("reinicie o ambiente e rode tudo de novo — a instalação ocorreu neste mesmo processo.")

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[key] = "2"
random.seed(42)
PINNED = {"numpy": "2.3.5", "pandas": "2.2.3", "scipy": "1.17.0",
          "scikit-learn": "1.8.0", "lightgbm": "4.6.0", "catboost": "1.2.8",
          "torch": "2.6.0", "tabm": "0.0.3", "rtdl-num-embeddings": "0.0.12",
          "threadpoolctl": "3.6.0", "joblib": "1.6.0"}
IN_COLAB = "google.colab" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG"))
BASE_DIR = Path(os.environ.get("GCI_WORKDIR", "/content/gci_home_credit_v21" if IN_COLAB else Path.cwd())).resolve()
BASE_DIR.mkdir(parents=True, exist_ok=True)
_COLAB_LOG_DIR = BASE_DIR / "output"
_COLAB_LOG_DIR.mkdir(exist_ok=True)
print("Python:", sys.version)
print("Plataforma:", platform.platform())
print("Pasta de trabalho:", BASE_DIR)
if (platform.python_implementation() != "CPython" or sys.version_info[:2] not in {(3,11), (3,12), (3,13)}
        or _colab_sysconfig.get_config_var("Py_GIL_DISABLED")):
    raise RuntimeError("Python não coberto pelos wheels auditados. No Colab, selecione o runtime 2026.07 (Python 3.12), reconecte e rode a primeira célula. Não altere os pins.")
if IN_COLAB and (sys.platform != "linux" or platform.machine().lower() not in {"x86_64", "amd64"}):
    raise RuntimeError("Esta preparação foi auditada para Colab Linux x86-64, CPU. Selecione esse tipo de ambiente.")

_installed = {}
for _name in PINNED:
    try:
        _installed[_name] = importlib.metadata.version(_name)
    except importlib.metadata.PackageNotFoundError:
        _installed[_name] = None
_wanted = dict(PINNED)
if sys.platform == "linux":
    _wanted["torch"] = "2.6.0+cpu"
_needed = [_name for _name,_version in _wanted.items()
           if _installed[_name] is None or (_installed[_name] != _version if _name == "torch" and sys.platform == "linux"
                                           else _installed[_name].split("+")[0] != _version)]
if _needed:
    # Uma transação, só wheels; nenhum import numérico antes do reinício.
    _V21_RESTART_REQUIRED = True
    _command = [sys.executable, "-m", "pip", "install", "--only-binary=:all:", "--upgrade-strategy", "only-if-needed",
                "--index-url", "https://pypi.org/simple"]
    if sys.platform == "linux":
        _command += ["--extra-index-url", "https://download.pytorch.org/whl/cpu"]
    _command += [f"{_name}=={_version}" for _name,_version in _wanted.items()]
    _install_record = dict(python=sys.version, platform=platform.platform(), previous_versions=_installed,
                           requested_versions=_wanted, status="restart_required", seconds=None)
    try:
        subprocess.run(_command, check=True)
    except subprocess.CalledProcessError as _error:
        _install_record["status"] = "installation_failed_restart_required"
        _install_record["seconds"] = _colab_time.perf_counter()-_COLAB_ENV_STARTED
        (_COLAB_LOG_DIR/"instalacao_colab.json").write_text(_colab_json.dumps(_install_record,indent=2)+"\n")
        raise RuntimeError("A instalação falhou; não continue com versões parciais. reinicie o ambiente e rode tudo de novo. Se não houver wheel, selecione o runtime 2026.07/Python 3.12, sem mudar os pins.") from _error
    _install_record["seconds"] = _colab_time.perf_counter()-_COLAB_ENV_STARTED
    (_COLAB_LOG_DIR/"instalacao_colab.json").write_text(_colab_json.dumps(_install_record,indent=2)+"\n")
    raise RuntimeError("Instalação concluída. reinicie o ambiente e rode tudo de novo. Use Ambiente de execução > Reiniciar sessão (Restart session), depois Executar tudo. Reexecutar só esta célula NÃO substitui o reinício.")

# Detectar módulos que ficaram em memória depois de uma instalação externa.
_module_packages = {"numpy":"numpy", "pandas":"pandas", "scipy":"scipy", "torch":"torch",
                    "sklearn":"scikit-learn", "lightgbm":"lightgbm", "catboost":"catboost"}
for _module,_package in _module_packages.items():
    _loaded = sys.modules.get(_module)
    _loaded_version = getattr(_loaded,"__version__",None) if _loaded is not None else None
    if _loaded_version is not None and (str(_loaded_version).split("+")[0] != PINNED[_package]
            or (_module == "torch" and sys.platform == "linux" and str(_loaded_version) != _wanted["torch"])):
        _V21_RESTART_REQUIRED = True
        raise RuntimeError(f"{_module} em memória difere do pin instalado: reinicie o ambiente e rode tudo de novo.")
assert {p:importlib.metadata.version(p).split("+")[0] for p in PINNED} == PINNED
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd()/".mpl-cache"))
import torch  # Primeiro import numérico, como na v21 validada.
_V21_ENV_READY = True
_COLAB_STAGE_TIMES = [{"etapa":"Ambiente/import inicial", "segundos":_colab_time.perf_counter()-_COLAB_ENV_STARTED}]
(_COLAB_LOG_DIR/"ambiente_colab.json").write_text(_colab_json.dumps(dict(
    python=sys.version, platform=platform.platform(), versions=_installed,
    requested_versions=_wanted, status="ready", restart_required=False),indent=2)+"\n")
print("Ambiente conferido. Versões base idênticas aos pins v21; prossiga para os dados.")
