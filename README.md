# An audit and restoration of the vector-sieve estimates for primes m^2 + n^2 + 1 with p + 2 almost prime

Ruqing Chen (2026). doi:[10.5281/zenodo.24237370](https://doi.org/10.5281/zenodo.24237370)

K. Nath and L. Xie (Acta Arith. 223 (2026), 51–82; arXiv:2501.16723v3) state that there are `>> x (log x)^(-5/2)`
primes `p <= x` with `p = m^2 + n^2 + 1` and `Omega(p+2) <= 9`. This note recomputes the numerical inequality
`H > 0` on which the proof ends.

* The linear sieve functions in the paper's Section 5 and in its Matlab files solve the delay system without the
  factor `1/s`; the computed lower function exceeds the computed upper function.
* With the correct functions, the reported `G(0.4230, 0.0249) = 0.001378` and `H(0.17, 0.25, 0.449, 0.011) = 1.2690`
  become `-23.73` and `-34.28`. The reported values are reproduced from the authors' code (`-0.0009`, `1.2625`).
* Three constants in the switched term (Proposition 4.2) are corrected, and its levels can be chosen more freely.
* With these changes the method gives `Omega(p+2) <= 17`; with the constants as printed, `Omega(p+2) <= 59`.
  With the Buchstab fixed point of Part XIV (one-sided rule): `<= 15` (15.947 at n = 1600, 15.809 at n = 3200).

The statements are conditional on Propositions 3.10, 4.1 and 4.3 of the Nath–Xie paper. The numerical values come
from floating-point grid searches, not interval arithmetic.

| path | content |
|---|---|
| `nath_xie_sieve_audit.tex`, `.pdf` | the note (9 pages) |
| `code/verify_nath_xie.py` | pure-Python recomputation of all tables (about 15 s; `--quick` for coarser grids) |
| `code/make_figures.py` | the figures (matplotlib) |
| `code/part14_fixed_point.py` | Remark 6.2 (numpy; needs the Part XIV engine `two-squares-part14/code` next to this repository; about 12 min) |
| `data/verify_nath_xie.txt`, `data/part14_fixed_point.txt` | outputs of the scripts |
| `data/part14_margin.txt` | n = 3200 check of the bound 15 (script `code/nath_xie_margin.py` in the Part XIV repository) |
