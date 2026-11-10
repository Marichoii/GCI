# Verificar o arquivo gravado, sem alterar seus bytes ou suas previsões.
def _colab_validate_submission(path, expected_ids):
    checked = pd.read_csv(path,float_precision="round_trip")
    if checked.shape != (61500,2):
        raise ValueError(f"CSV inválido: formato {checked.shape}; esperado (61500,2).")
    if checked.columns.tolist() != ["SK_ID_CURR","TARGET"]:
        raise ValueError("CSV inválido: colunas devem ser SK_ID_CURR,TARGET, nessa ordem.")
    if checked.isna().any().any():
        raise ValueError("CSV inválido: há NaN em IDs ou previsões.")
    if not np.array_equal(checked.SK_ID_CURR.to_numpy(),np.asarray(expected_ids)):
        raise ValueError("CSV inválido: IDs não estão na ordem de input/test.csv.")
    values = checked.TARGET.to_numpy()
    if not np.issubdtype(values.dtype,np.number) or not np.isfinite(values).all():
        raise ValueError("CSV inválido: TARGET deve ser numérico e finito.")
    if not ((values>=0)&(values<=1)).all():
        raise ValueError("CSV inválido: TARGET fora do intervalo [0,1].")
    return dict(linhas=len(checked),colunas=checked.columns.tolist(),ids_na_ordem=True,
                sem_nan=True,valores_entre_zero_e_um=True,sha256=_colab_sha256_file(path))

def _colab_time_warning(total_seconds):
    exceeded = total_seconds > 90*60
    if exceeded:
        print("AVISO: o tempo total desta execução passou de 90 minutos. O aviso não muda nem interrompe o treinamento.")
    return exceeded

_COLAB_MAIN_SECONDS = (_COLAB_MAIN_FINISHED_AT or _colab_time.perf_counter())-_COLAB_MAIN_STARTED
_COLAB_VERIFY_STARTED = _colab_time.perf_counter()
assert RUN_CV is False and CV_ONLY is False and recipe == FROZEN_V21, "A receita congelada não foi preservada."
if csv_sha256 is None or full_train_seconds is None:
    raise RuntimeError("O treino integral não produziu CSV nesta sessão; não valide saídas antigas como resultado novo.")
_colab_check_inputs(BASE_DIR/"input")
_fresh_test_ids = pd.read_csv(BASE_DIR/"input/test.csv",usecols=["SK_ID_CURR"]).SK_ID_CURR.to_numpy()
_validation = _colab_validate_submission(output_file,_fresh_test_ids)
assert _validation["sha256"] == csv_sha256, "O CSV foi alterado depois de gravado."
_historical_sha = "48334efc455a9a28f395e945fb4c04ee847458f439d8b26984e3965766d6a756"
_validation["sha256_historico_macos"] = _historical_sha
_validation["identico_ao_historico_macos"] = _validation["sha256"] == _historical_sha
print("CSV conferido:",output_file)
print("SHA-256:",_validation["sha256"])
if not _validation["identico_ao_historico_macos"]:
    print("ATENÇÃO: este SHA-256 difere do histórico macOS. As verificações de formato passaram, mas isso não comprova identidade entre Colab/Linux e macOS. Preserve e registre ambos os arquivos; não ajuste previsões para forçar o hash.")
_COLAB_STAGE_TIMES.append({"etapa":"Leitura final, preparo/auxiliares e montagem", "segundos":max(0.0,_COLAB_MAIN_SECONDS-full_train_seconds)})
_COLAB_STAGE_TIMES += [{"etapa":entry["component"],"segundos":entry["seconds"]} for entry in full_timings]
_COLAB_STAGE_TIMES.append({"etapa":"Agregação, gravação e sobrecarga entre componentes", "segundos":max(0.0,full_train_seconds-sum(entry["seconds"] for entry in full_timings))})
_COLAB_STAGE_TIMES.append({"etapa":"Verificação do CSV e hashes de entrada", "segundos":_colab_time.perf_counter()-_COLAB_VERIFY_STARTED})
_colab_total_seconds = _colab_time.perf_counter()-_COLAB_RUN_STARTED
_colab_exceeded_90 = _colab_time_warning(_colab_total_seconds)
print(pd.DataFrame(_COLAB_STAGE_TIMES).assign(minutos=lambda frame:frame.segundos/60).to_string(index=False))
print(f"Total desde a primeira célula após reinício: {_colab_total_seconds/60:.2f} minutos (inclui entrada e esperas entre células).")
_colab_final_report = dict(validacao=_validation,etapas=_COLAB_STAGE_TIMES,total_seconds=_colab_total_seconds,
                           exceeded_90_minutes=_colab_exceeded_90,python=sys.version,platform=platform.platform(),
                           timing_mode=_COLAB_TIMING_MODE,
                           installed_versions={p:importlib.metadata.version(p) for p in PINNED},
                           threads_per_model=2,recipe_unchanged=True,training_report="execution_report.json")
(output_dir/"verificacao_colab.json").write_text(_colab_json.dumps(_colab_final_report,ensure_ascii=False,indent=2)+"\n")
(output_dir/"SHA256.txt").write_text(_validation["sha256"]+"  submission.csv\n")
