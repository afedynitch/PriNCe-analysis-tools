# Auger FD Phase-I 2026 Xmax-scale systematic

From the data release of Pierre Auger Collaboration, "Depth of Maximum of Air-Shower Profiles above 10^17.7 eV
Measured with the Fluorescence Detector of the Pierre Auger Observatory" (arXiv:2605.12598), Zenodo
10.5281/zenodo.18024023, files unchanged:

- `xmaxSystematics.txt`: energy bin, lg(E/eV) bin edges, upper and lower systematic uncertainty of the Xmax
  scale in g/cm^2.
- `correlationSysPlus.txt`, `correlationSysMinus.txt`: 20 x 20 bin-to-bin correlation of that uncertainty for
  upward and downward shifts (the release README says 18 x 18).

The moments themselves are the table `_auger2026FD_table` in `prince_analysis_tools/spectra.py`.
