"""Test Colab operational cells without running any fit/train function."""
import ast
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace, ModuleType
import unittest
import warnings
from unittest import mock

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OLD_PATH=ROOT.parent/"entregaveis/v21/gci_home_credit_v21.ipynb"
NB=json.loads((ROOT/"gci_home_credit_v21_colab.ipynb").read_text())
OLD=json.loads(OLD_PATH.read_text())
CELLS={c.get("metadata",{}).get("v21_colab_role"):"".join(c["source"]) for c in NB["cells"] if c["cell_type"]=="code" and c.get("metadata",{}).get("v21_colab_role")}
PINS={line.split("==")[0]:line.split("==")[1] for line in (OLD_PATH.parent/"requirements.txt").read_text().splitlines()}
EXPECTED_CSV_SHA="48334efc455a9a28f395e945fb4c04ee847458f439d8b26984e3965766d6a756"
PRESERVE=[OLD_PATH,OLD_PATH.parent/"gci_home_credit_v21.zip",OLD_PATH.parent/"output/submission.csv",
          ROOT/"gci_home_credit_v21.ipynb",ROOT/"gci_home_credit_v21.zip",ROOT/"input/train.csv",ROOT/"input/test.csv"]
BEFORE={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in PRESERVE}


def assigned(source,name):
    return next(ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))


def configured(source,**values):
    tree=ast.parse(source)
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in values:
                    node.value=ast.parse(repr(values[target.id]),mode="eval").body
    return compile(ast.fix_missing_locations(tree),"configured_input_cell","exec")


@contextlib.contextmanager
def fake_pip(side_effect=None):
    # Intercept only pip. platform/Torch may use subprocess for system probes;
    # do not capture their environment dictionaries in mock assertion output.
    original_run=subprocess.run
    captured=mock.Mock(side_effect=side_effect)
    def run(command,*args,**kwargs):
        if isinstance(command,list) and command[1:3]==["-m","pip"]:
            return captured(command,*args,**kwargs)
        return original_run(command,*args,**kwargs)
    with mock.patch("subprocess.run",side_effect=run):
        yield captured


class ColabCells(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="gci_v21_colab_tests_")
        self.base=Path(self.temp.name)
        self.ns={"__name__":"v21_operational_tests"}

    def tearDown(self):
        self.temp.cleanup()

    def environment(self,namespace=None):
        target=namespace if namespace is not None else self.ns
        with mock.patch.dict(os.environ,{"GCI_WORKDIR":str(self.base)}),contextlib.redirect_stdout(io.StringIO()):
            exec(CELLS["ambiente"],target)
        return target

    def inputs(self,mode="existente",**options):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(configured(CELLS["entrada"],INPUT_MODE=mode,**options),self.ns)

    def real_inputs(self):
        self.environment()
        (self.base/"input").symlink_to(ROOT/"input",target_is_directory=True)
        self.inputs()

    def validator(self):
        self.real_inputs()
        self.ns.update(np=np,pd=pd)
        for node in ast.parse(CELLS["verificacao"]).body:
            if isinstance(node,ast.FunctionDef):
                exec(compile(ast.Module(body=[node],type_ignores=[]),"verification_functions","exec"),self.ns)
        return self.ns["_colab_validate_submission"]

    def test_model_cells_and_frozen_identical(self):
        for old_cell in OLD["cells"][2:]:
            self.assertEqual(sum(cell==old_cell for cell in NB["cells"]),1)
        old_main="".join(OLD["cells"][26]["source"])
        new_main=next("".join(c["source"]) for c in NB["cells"] if c==OLD["cells"][26])
        self.assertEqual(old_main,new_main)
        self.assertEqual(assigned(old_main,"FROZEN_V21"),assigned(new_main,"FROZEN_V21"))
        self.assertEqual(assigned(CELLS["ambiente"],"PINNED"),PINS)

    def test_actual_installed_environment_no_install(self):
        with fake_pip() as pip:
            self.environment()
        pip.assert_not_called()
        self.assertTrue(self.ns["_V21_ENV_READY"])

    def test_notebook_schema_without_new_missing_cell_ids(self):
        import nbformat
        counts=[]
        for notebook in [OLD,NB]:
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter('always')
                nbformat.validate(copy.deepcopy(notebook))
            counts.append(sum('missing an id' in str(w.message) for w in observed))
        self.assertEqual(counts[0],counts[1])
        for cell in NB['cells']:
            if cell.get('metadata',{}).get('v21_colab_role'):self.assertIn('id',cell)

    def test_install_then_restart_and_rerun_still_stops(self):
        real_version=__import__("importlib.metadata",fromlist=["version"]).version
        def version(name):return "0.0.0" if name=="numpy" else real_version(name)
        with mock.patch("importlib.metadata.version",side_effect=version),fake_pip() as pip:
            with self.assertRaisesRegex(RuntimeError,"reinicie o ambiente e rode tudo de novo"):
                self.environment()
            self.assertEqual(pip.call_count,1)
            self.assertIn("--only-binary=:all:",pip.call_args.args[0])
            self.assertIn("numpy==2.3.5",pip.call_args.args[0])
            with self.assertRaisesRegex(RuntimeError,"reinicie o ambiente e rode tudo de novo"):
                self.environment()
            self.assertEqual(pip.call_count,1)
        self.assertFalse(self.ns["_V21_ENV_READY"])

    def test_failed_install_stops_partial_environment(self):
        with mock.patch("importlib.metadata.version",return_value="0.0.0"),fake_pip(side_effect=subprocess.CalledProcessError(1,["pip"])):
            with self.assertRaisesRegex(RuntimeError,"instalação falhou"):
                self.environment()
        self.assertFalse(self.ns["_V21_ENV_READY"])
        self.assertTrue(self.ns["_V21_RESTART_REQUIRED"])

    def test_loaded_numpy_mismatch_stops(self):
        with mock.patch.object(np,"__version__","0.0.0"),fake_pip() as pip:
            with self.assertRaisesRegex(RuntimeError,"em memória"):
                self.environment()
        pip.assert_not_called()

    def test_uncovered_python_stops_without_install(self):
        with mock.patch.object(sys,"version_info",(3,14,0,"final",0)),fake_pip() as pip:
            with self.assertRaisesRegex(RuntimeError,"Python não coberto"):
                self.environment()
        pip.assert_not_called()

    def test_loaded_torch_build_mismatch_stops(self):
        import torch
        real_version=__import__("importlib.metadata",fromlist=["version"]).version
        def version(name):return "2.6.0+cpu" if name=="torch" else real_version(name)
        with mock.patch.object(sys,"platform","linux"),mock.patch("importlib.metadata.version",side_effect=version),mock.patch.object(torch,"__version__","2.6.0+cu124"),fake_pip() as pip:
            with self.assertRaisesRegex(RuntimeError,"em memória"):
                self.environment()
        pip.assert_not_called()

    def test_linux_requests_cpu_build(self):
        real_version=__import__("importlib.metadata",fromlist=["version"]).version
        def version(name):return "2.6.0+cu124" if name=="torch" else real_version(name)
        with mock.patch.object(sys,"platform","linux"),mock.patch("importlib.metadata.version",side_effect=version),fake_pip() as pip:
            with self.assertRaisesRegex(RuntimeError,"reinicie"):
                self.environment()
            self.assertIn("torch==2.6.0+cpu",pip.call_args.args[0])
            self.assertIn("https://download.pytorch.org/whl/cpu",pip.call_args.args[0])

    def test_real_inputs_match_readme_without_copy(self):
        self.real_inputs()
        readme=(ROOT/"input/README.md").read_text()
        for digest in self.ns["_input_hashes_checked"].values():self.assertIn(digest,readme)

    def test_bad_input_stops(self):
        self.environment();(self.base/"input").mkdir()
        for name in ["train.csv","test.csv"]:(self.base/"input"/name).write_bytes(b"wrong-data\n")
        with self.assertRaisesRegex(ValueError,"SHA-256 incorreto"):self.inputs()

    def test_missing_existing_input_stops(self):
        self.environment()
        with self.assertRaises(FileNotFoundError):self.inputs()

    def test_mock_upload_only_correct_files_copied(self):
        self.environment();self.ns["IN_COLAB"]=True
        files=SimpleNamespace(upload=mock.Mock(return_value={n:(ROOT/"input"/n).read_bytes() for n in ["train.csv","test.csv"]}))
        fake=ModuleType("google.colab");fake.files=files
        with mock.patch.dict(sys.modules,{"google.colab":fake}):self.inputs(mode="upload")
        files.upload.assert_called_once()
        self.assertEqual(self.ns["_colab_check_inputs"](self.base/"input"),self.ns["EXPECTED_INPUT_SHA256"])

    def test_mock_upload_bad_hash_not_copied(self):
        self.environment();self.ns["IN_COLAB"]=True
        fake=ModuleType("google.colab");fake.files=SimpleNamespace(upload=lambda:{"train.csv":b"bad","test.csv":b"bad"})
        with mock.patch.dict(sys.modules,{"google.colab":fake}),self.assertRaisesRegex(ValueError,"SHA-256 incorreto"):
            self.inputs(mode="upload")
        self.assertEqual(list((self.base/"input").iterdir()),[])

    def test_mock_drive_mount_and_copy(self):
        self.environment();self.ns["IN_COLAB"]=True
        mount=mock.Mock();fake=ModuleType("google.colab");fake.drive=SimpleNamespace(mount=mount)
        with mock.patch.dict(sys.modules,{"google.colab":fake}):self.inputs(mode="drive",DRIVE_INPUT_DIR=str(ROOT/"input"))
        mount.assert_called_once_with("/content/drive")
        self.assertEqual(self.ns["_colab_check_inputs"](self.base/"input"),self.ns["EXPECTED_INPUT_SHA256"])

    def test_research_environment_blocked(self):
        self.real_inputs()
        with mock.patch.dict(os.environ,{"GCI_RUN_CV":"1"}),self.assertRaisesRegex(RuntimeError,"pesquisa está encerrada"):
            exec(CELLS["antes_do_treino"],self.ns)

    def test_existing_csv_not_overwritten(self):
        self.real_inputs();target=self.base/"output/submission.csv";target.write_bytes(b"preserve")
        with mock.patch.dict(os.environ,{"GCI_RUN_CV":"0","GCI_CV_ONLY":"0"}),self.assertRaises(FileExistsError):
            exec(CELLS["antes_do_treino"],self.ns)
        self.assertEqual(target.read_bytes(),b"preserve")

    def test_timing_hooks_without_executing_model_cell(self):
        self.real_inputs()
        callbacks={}
        events=SimpleNamespace(register=lambda e,c:callbacks.__setitem__(e,c),unregister=lambda e,c:callbacks.pop(e))
        self.ns["get_ipython"]=lambda:SimpleNamespace(events=events)
        with mock.patch.dict(os.environ,{"GCI_RUN_CV":"0","GCI_CV_ONLY":"0"}),contextlib.redirect_stdout(io.StringIO()):
            exec(CELLS["antes_do_treino"],self.ns)
        other=SimpleNamespace(raw_cell="1+1")
        callbacks["post_run_cell"](SimpleNamespace(info=other))
        self.assertIsNone(self.ns["_COLAB_MAIN_FINISHED_AT"])
        info=SimpleNamespace(raw_cell="".join(OLD["cells"][26]["source"]))
        callbacks["pre_run_cell"](info)
        callbacks["post_run_cell"](SimpleNamespace(info=info))
        self.assertGreaterEqual(self.ns["_COLAB_MAIN_FINISHED_AT"],self.ns["_COLAB_MAIN_STARTED"])

    def test_historical_csv_passes_read_only(self):
        validator=self.validator()
        ids=pd.read_csv(ROOT/"input/test.csv",usecols=["SK_ID_CURR"]).SK_ID_CURR.to_numpy()
        result=validator(OLD_PATH.parent/"output/submission.csv",ids)
        self.assertEqual(result["sha256"],EXPECTED_CSV_SHA)
        self.assertEqual(result["linhas"],61500)

    def test_csv_invalid_cases_rejected_without_writing_predictions(self):
        validator=self.validator();path=OLD_PATH.parent/"output/submission.csv"
        valid=pd.read_csv(path,float_precision="round_trip");ids=valid.SK_ID_CURR.to_numpy().copy()
        cases={"rows":valid.iloc[:-1].copy(),"columns":valid.rename(columns={"TARGET":"WRONG"}),
               "order":valid.iloc[::-1].reset_index(drop=True),"nan":valid.copy(),
               "infinite":valid.copy(),"range":valid.copy(),"text":valid.astype({"TARGET":str})}
        cases["nan"].loc[0,"TARGET"]=np.nan
        cases["infinite"].loc[0,"TARGET"]=np.inf
        cases["range"].loc[0,"TARGET"]=1.1
        cases["text"].loc[0,"TARGET"]="bad"
        for label,frame in cases.items():
            with self.subTest(label=label),mock.patch.object(pd,"read_csv",return_value=frame),self.assertRaises(ValueError):
                validator(path,ids)

    def test_time_warning_threshold_without_waiting(self):
        self.validator();warn=self.ns["_colab_time_warning"]
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertFalse(warn(5399));self.assertFalse(warn(5400));self.assertTrue(warn(5401))
        self.assertIn("90 minutos",output.getvalue())


if __name__=="__main__":
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ColabCells))
    print(stream.getvalue())
    after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in PRESERVE}
    assert after==BEFORE,"An existing validated artifact changed."
    report=dict(tests_run=result.testsRun,passed=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),
        training_executed=False,colab_executed=False,real_install_executed=False,
        upload_drive_tests="local mocks, not authenticated transfers",historical_csv_read_only=True,
        historical_artifacts_unchanged=after,notebook_sha256=hashlib.sha256((ROOT/"gci_home_credit_v21_colab.ipynb").read_bytes()).hexdigest())
    (ROOT/"v21_colab/evidence/test_report.json").write_text(json.dumps(report,indent=2)+"\n")
    (ROOT/"v21_colab/evidence/test_console.log").write_text(stream.getvalue())
    if not result.wasSuccessful():raise SystemExit(1)
