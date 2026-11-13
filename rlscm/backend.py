"""Persistent MATLAB SCM; coordinates and solver physics stay unchanged."""
import os,subprocess,time,tempfile,shutil
from pathlib import Path
import numpy as np
from scipy.io import loadmat,savemat
from .paths import ROOT,LOW,HIGH,matlab_executable
from .coordinates import rasterize_physical
from .scm_solver import solve_scm_dispersion

class SCM:
    def __init__(self, folder, nodes=30, frequencies=601, curve_points=0):
        self.archive = folder / 'worker'
        self.archive.mkdir(parents=True,exist_ok=True)
        self.folder = Path(tempfile.mkdtemp(prefix='scm_rl_'))
        self.nodes = nodes
        self.frequencies = frequencies
        self.curve_points = curve_points
        self.f = np.linspace(1e3, 2e6, frequencies)
        self.times = []
        self.log = open(self.folder / 'matlab.log', 'w', encoding='utf-8')
        started = time.perf_counter()
        backend_pref = os.environ.get('RLSCM_BACKEND', '').strip().lower()
        executable = None if backend_pref == 'python' else matlab_executable()
        if not executable:
            self.use_python = True
            self.process = None
            self.log.write('Using native Python/SciPy LAPACK SCM solver (Colab / no-MATLAB mode).\n')
            self.log.flush()
            self.startup = time.perf_counter() - started
            return
        self.use_python = False
        quote = lambda p: str(p).replace("'", "''").replace('\\', '/')
        command = f"addpath('{quote(ROOT / 'solver')}');rl_worker('{quote(self.folder)}')"
        self.process = subprocess.Popen(
            [executable, '-singleCompThread', '-batch', command],
            stdout=self.log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        self.wait(self.folder / 'ready.mat', 180)
        self.startup = time.perf_counter() - started

    def wait(self, path, timeout=120):
        started = time.perf_counter()
        while not path.exists():
            if self.process.poll() is not None:
                raise RuntimeError(f'MATLAB exited; see {self.folder / "matlab.log"}')
            if time.perf_counter() - started > timeout:
                self.close()
                raise TimeoutError(f'MATLAB timeout: {path}')
            time.sleep(.01)

    def solve(self, m):
        m = np.asarray(m, dtype=float)
        if m.shape != (3,) or not np.isfinite(m).all() or np.any(m < LOW-1e-12) or np.any(m > HIGH+1e-12):
            raise ValueError('Parameters outside the b8 search bounds')
        m = np.clip(m, LOW, HIGH)
        start = time.perf_counter()
        if getattr(self, 'use_python', False):
            data = solve_scm_dispersion(m, nodes=self.nodes, frequencies=len(self.f), curve_points=self.curve_points)
            result = {k: np.atleast_1d(data[k]) for k in ['ff', 'kk', 'bins']}
            if self.curve_points:
                curve_k = np.atleast_1d(data['curve_k'])
                curve_f = np.asarray(data['curve_f']).reshape(-1, self.curve_points)
                valid = np.isfinite(curve_f)
                result['curve_k'], result['curve_f'] = curve_k, curve_f
                result['image'] = rasterize_physical(curve_f[valid],
                    np.broadcast_to(curve_k, curve_f.shape)[valid])[0]
            self.times.append(time.perf_counter() - start)
            return result
        temp = self.folder / 'request.tmp.mat'
        savemat(temp, dict(m=m, N=self.nodes, f_eval=self.f, n_k=self.curve_points))
        for attempt in range(100):
            try:
                temp.replace(self.folder / 'request.mat')
                break
            except OSError:
                if attempt == 99:
                    raise
                time.sleep(.02)
        response = self.folder / 'response.mat'
        self.wait(response)
        for attempt in range(100):
            try:
                data = loadmat(response, squeeze_me=True)
                break
            except Exception:
                if attempt == 99:
                    raise
                time.sleep(.02)
        for attempt in range(100):
            try:
                response.unlink(missing_ok=True)
                break
            except OSError:
                if attempt == 99:
                    raise
                time.sleep(.02)
        if 'error_message' in data:
            raise RuntimeError(str(data['error_message']))
        result = {k: np.atleast_1d(data[k]) for k in ['ff', 'kk', 'bins']}
        if not len(result['kk']):
            raise RuntimeError('SCM returned no propagating roots')
        if self.curve_points:
            curve_k = np.atleast_1d(data['curve_k'])
            curve_f = np.asarray(data['curve_f']).reshape(-1, self.curve_points)
            valid = np.isfinite(curve_f)
            if not valid.any():
                raise RuntimeError('SCM returned no dense dispersion curves')
            result['curve_k'], result['curve_f'] = curve_k, curve_f
            result['image'] = rasterize_physical(curve_f[valid],
                np.broadcast_to(curve_k, curve_f.shape)[valid])[0]
        self.times.append(time.perf_counter() - start)
        return result

    def close(self):
        if getattr(self, 'process', None) is not None:
            (self.folder / 'stop').touch()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.log.close()
        self.archive.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.folder / 'matlab.log', self.archive / 'matlab.log')

