# Opcional: se input/ já está correto, a célula apenas confere os hashes.
INPUT_MODE = "upload" # @param ["upload", "drive", "existente"]
DRIVE_INPUT_DIR = "/content/drive/MyDrive/GCI/input" # @param {type:"string"}

assert globals().get("_V21_ENV_READY",False), "Execute a primeira célula e reinicie se solicitado."
import hashlib as _colab_hashlib
import shutil as _colab_shutil
import tempfile as _colab_tempfile
_COLAB_INPUT_STARTED = _colab_time.perf_counter()
EXPECTED_INPUT_SHA256 = {
    "train.csv":"b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1",
    "test.csv":"dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65"}

def _colab_sha256_file(path):
    digest = _colab_hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()

def _colab_check_inputs(directory):
    actual = {}
    for name,expected in EXPECTED_INPUT_SHA256.items():
        path = Path(directory)/name
        if not path.is_file():
            raise FileNotFoundError(f"Falta {path}. Use upload ou Drive com os dois CSVs oficiais do GCI.")
        actual[name] = _colab_sha256_file(path)
        if actual[name] != expected:
            raise ValueError(f"SHA-256 incorreto em {path}: {actual[name]}; esperado {expected}. Pare e corrija o arquivo; não use dados do Kaggle.")
    return actual

_input_dir = BASE_DIR/"input"
_input_dir.mkdir(exist_ok=True)
_all_present = all((_input_dir/name).is_file() for name in EXPECTED_INPUT_SHA256)
if _all_present:
    _input_hashes_checked = _colab_check_inputs(_input_dir)
    print("input/ já existe e foi validado; nenhum upload/cópia necessário.")
else:
    if INPUT_MODE == "upload":
        if not IN_COLAB:
            raise RuntimeError("Upload só está disponível no Colab. Localmente use INPUT_MODE='existente' e prepare input/.")
        from google.colab import files
        # O upload grava arquivos; isolá-los antes de validar e copiar.
        _old_cwd = Path.cwd()
        with _colab_tempfile.TemporaryDirectory(prefix="gci_v21_upload_",dir=BASE_DIR) as _staging:
            try:
                os.chdir(_staging)
                print("Selecione juntos train.csv e test.csv, com esses nomes exatos.")
                _uploaded = files.upload()
            finally:
                os.chdir(_old_cwd)
            if set(_uploaded) != set(EXPECTED_INPUT_SHA256):
                raise ValueError("Selecione somente train.csv e test.csv, juntos, sem renomear os arquivos.")
            for _name,_data in _uploaded.items():
                (Path(_staging)/_name).write_bytes(_data)
            _colab_check_inputs(_staging)
            for _name in EXPECTED_INPUT_SHA256:
                _colab_shutil.copyfile(Path(_staging)/_name,_input_dir/_name)
        del _uploaded
    elif INPUT_MODE == "drive":
        if not IN_COLAB:
            raise RuntimeError("Montagem de Drive só está disponível no Colab.")
        from google.colab import drive
        drive.mount("/content/drive")
        _source = Path(DRIVE_INPUT_DIR)
        _colab_check_inputs(_source)
        for _name in EXPECTED_INPUT_SHA256:
            if (_source/_name).resolve() != (_input_dir/_name).resolve():
                _colab_shutil.copyfile(_source/_name,_input_dir/_name)
    elif INPUT_MODE != "existente":
        raise ValueError("INPUT_MODE deve ser upload, drive ou existente.")
    _input_hashes_checked = _colab_check_inputs(_input_dir)
_COLAB_STAGE_TIMES.append({"etapa":"Entrada: upload/cópia e hashes", "segundos":_colab_time.perf_counter()-_COLAB_INPUT_STARTED})
for _name,_digest in _input_hashes_checked.items():
    print(_name,"SHA-256",_digest)
