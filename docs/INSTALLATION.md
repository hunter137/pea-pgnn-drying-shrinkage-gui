# Installation and startup

## Windows quick start

1. Install 64-bit Python 3.9 or later.
2. Clone or download this repository.
3. Open PowerShell in the repository folder.
4. Create an isolated environment and install the locked dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

5. Start the application:

   ```powershell
   .\.venv\Scripts\python.exe run_gui.py
   ```

After installation, `launch_gui.bat` can be double-clicked when the active
Python installation contains the required packages.

## Optional MathType integration

Desktop MathType is optional. When installed in its standard Windows location,
the formula editor can send the current equation to MathType and read back its
structured MathML clipboard representation. All other formula editing,
conversion, preview, and trial-calculation features work without MathType.

## GPU use

Inference automatically uses CUDA when the installed PyTorch build reports an
available compatible GPU; otherwise it runs on the CPU. Follow PyTorch's
official installation selector if a CUDA-specific build is required for your
machine. The bundled weights are identical for CPU and GPU inference.

## Training data

The trained deployment artifacts are bundled for inference. Retraining and the
strict audit require the private frozen research database and split files.
Update `configs/training.json` to point to authorized local copies. Those data
are intentionally excluded from Git.

