from pathlib import Path
import os
import shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
DATA=PROJECT/"data"/"bone_b8_train_256_compact.h5"
LOW=np.array([.5,1.5,.15])
HIGH=np.array([5.5,4.,.42])

def matlab_executable():
    configured=os.environ.get('MATLAB_EXE') or shutil.which('matlab')
    if configured:
        return configured if Path(configured).is_file() else None
    installed=sorted((Path(os.environ.get('ProgramFiles',r'C:\Program Files'))/'MATLAB').glob('R*/bin/matlab.exe'),reverse=True)
    return str(installed[0]) if installed else None
