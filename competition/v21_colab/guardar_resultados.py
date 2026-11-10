# Opcional: só resultados verificados desta sessão, nunca modelos/checkpoints.
BAIXAR_RESULTADOS = True # @param {type:"boolean"}
COPIAR_PARA_DRIVE = False # @param {type:"boolean"}
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/GCI/resultados_v21" # @param {type:"string"}

import zipfile as _colab_zipfile
import datetime as _colab_datetime
assert globals().get("_validation",{}).get("sha256") == _colab_sha256_file(output_file), "Execute a verificação final antes de guardar os resultados."
_results_zip = output_dir/"resultados_v21_colab.zip"
with _colab_zipfile.ZipFile(_results_zip,"w",compression=_colab_zipfile.ZIP_DEFLATED) as _bundle:
    for _name in ["submission.csv","execution_report.json","verificacao_colab.json","ambiente_colab.json","instalacao_colab.json","SHA256.txt"]:
        if (output_dir/_name).is_file():
            _bundle.write(output_dir/_name,_name)
if COPIAR_PARA_DRIVE:
    if not IN_COLAB:
        raise RuntimeError("Cópia automática para Drive só está disponível no Colab.")
    from google.colab import drive
    drive.mount("/content/drive")
    _run_stamp = _colab_datetime.datetime.now(_colab_datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _drive_destination = Path(DRIVE_OUTPUT_DIR)/_run_stamp
    _drive_destination.mkdir(parents=True,exist_ok=False)
    _colab_shutil.copyfile(_results_zip,_drive_destination/_results_zip.name)
    print("Resultados copiados para",_drive_destination)
if BAIXAR_RESULTADOS and IN_COLAB:
    from google.colab import files
    try:
        files.download(str(_results_zip))
    except Exception as _download_error:
        print("Download automático não concluído:",_download_error,". Baixe o ZIP pelo painel Arquivos; o CSV e a verificação continuam salvos no runtime.")
print("Pacote de resultados:",_results_zip)
