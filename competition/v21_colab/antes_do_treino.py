# Guardas de ambiente/entrada. A célula seguinte é a original v21, inteira.
assert globals().get("_V21_ENV_READY",False), "Ambiente ainda não validado. Reinicie se solicitado e execute tudo."
if os.environ.get("GCI_RUN_CV","0") != "0" or os.environ.get("GCI_CV_ONLY","0") != "0":
    raise RuntimeError("A pesquisa está encerrada. Use GCI_RUN_CV=0 e GCI_CV_ONLY=0; somente FROZEN_V21 será reproduzido.")
_colab_check_inputs(BASE_DIR/"input")
if (BASE_DIR/"output/submission.csv").exists():
    raise FileExistsError("Já existe output/submission.csv. Para conferir/baixar, execute apenas as células finais se o treino já terminou nesta sessão. Para repetir do zero, use uma pasta GCI_WORKDIR nova; não sobrescreva uma execução anterior.")
_COLAB_MAIN_STARTED = _colab_time.perf_counter()
_COLAB_MAIN_FINISHED_AT = None
_COLAB_TIMING_MODE = "fallback: medição entre células, inclui possíveis esperas"
_COLAB_ORIGINAL_MAIN_SHA256 = "__ORIGINAL_MAIN_SHA256__"

def _colab_main_pre(info):
    global _COLAB_MAIN_STARTED
    if _colab_hashlib.sha256(info.raw_cell.strip().encode()).hexdigest() == _COLAB_ORIGINAL_MAIN_SHA256:
        _COLAB_MAIN_STARTED = _colab_time.perf_counter()

def _colab_main_post(result):
    global _COLAB_MAIN_FINISHED_AT
    if _colab_hashlib.sha256(result.info.raw_cell.strip().encode()).hexdigest() == _COLAB_ORIGINAL_MAIN_SHA256:
        _COLAB_MAIN_FINISHED_AT = _colab_time.perf_counter()
        if _COLAB_MAIN_FINISHED_AT-_COLAB_RUN_STARTED > 90*60:
            print("AVISO: o tempo total desta execução passou de 90 minutos; modelos e receita permaneceram iguais.")

try:
    _colab_ipy = get_ipython()
except NameError:
    _colab_ipy = None
if _colab_ipy is not None:
    # Eventos de célula, sem decorar, reescrever ou instrumentar modelos.
    for _event,_callback in globals().get("_COLAB_TIMING_HOOKS",[]):
        _colab_ipy.events.unregister(_event,_callback)
    _COLAB_TIMING_HOOKS = [("pre_run_cell",_colab_main_pre),("post_run_cell",_colab_main_post)]
    for _event,_callback in _COLAB_TIMING_HOOKS:
        _colab_ipy.events.register(_event,_callback)
    _COLAB_TIMING_MODE = "IPython pre/post_run_cell: duração da célula original"
print("Iniciando a célula original de treino integral. Modelos e FROZEN_V21 não foram alterados.")
