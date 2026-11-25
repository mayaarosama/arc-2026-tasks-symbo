That message means that device is missing one or both of the things the cut-out needs. The upload box has two different messages, and you may have seen parts of each:

| Message | What's missing | Fix |
| --- | --- | --- |
| "The cut-out model is not installed here… (`python -m scripts.common.download_weights --cutout`)" | the model file `weights/u2net/u2net.onnx` | download it |
| "The cut-out needs onnxruntime: `pip install -r requirements/conductor.txt`" | the `onnxruntime` package in the Python environment the conductor runs from | install the conductor requirements |

The conductor setup script fixes both in one go. It needs network access, and you run it on the device running the conductor.

**Linux / the workstation**
```bash
cd demo
scripts/cpu/cpu_box_linux.sh conductor
# or by hand:
.venv/bin/pip install -r requirements/conductor.txt
.venv/bin/python -m scripts.common.download_weights --cutout
.venv/bin/python -m scripts.common.verify_offline --cutout
```

**Windows**
```powershell
cd demo
powershell -ExecutionPolicy Bypass -File scripts\cpu\cpu_box_windows.ps1 conductor
```

Then:
1. **Install into the same environment the conductor runs from** (`.venv`). Installing into the system Python gives the same error again.
2. Restart the conductor to be safe, then **reload the operator page**. The "not installed" message is checked only when the page loads, so it stays until you reload.
3. `verify_offline.py --cutout` should pass, and uploading `IMG_2089.jpeg` should show *"Ready, with a warning…"*, as it did on this laptop.
