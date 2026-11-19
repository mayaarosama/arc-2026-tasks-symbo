# Lockfiles

One lock per worker env, produced on the DGX and installed identically everywhere else.
A different diffusers version between machines silently changes outputs and breaks the comparison.

    # on the DGX, after a clean install that passes verify_offline.py
    pip freeze > locks/t2i-gpu.lock.txt
    # on the workstation
    pip install -r locks/t2i-gpu.lock.txt
    # on the CPU box: same lock minus torch/nvidia-* lines (CPU wheel of the same torch version)

The conductor compares a hash of each worker's installed packages (torch and CUDA wheels excluded)
and shows a warning on the operator page when two machines differ for the same task.
