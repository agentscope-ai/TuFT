"""Trinity-rft vLLM compatibility patches for TuFT.

These patches fix incompatibilities between trinity-rft and vLLM >= 0.20.0.
Since trinity's patched code runs inside Ray remote actors (separate processes),
the patches must be installed into site-packages to take effect.

Run `python -m tuft.patches.apply` to install/re-install patches after
upgrading trinity-rft or vLLM.
"""
