# k-cut Reduced Shear

This module can be used to calculate the shear angular power spectrum,
reduced shear correction, Bernardeau-Nishimichi-Taruya transform, as 
well as make Fisher and bias forecasts while applying k-cut cosmic shear 
for Stage IV cosmic shear surveys, and a hypothetical kinematic lensing survey.

The main file kcut_rs.py contains classes which handle these.

The module uses the [BiHalofit](http://cosmo.phys.hirosaki-u.ac.jp/takahasi/codes_e.htm) (
[Takahashi et al., 2020](https://arxiv.org/abs/1911.07886)), [BNT transform](https://github.com/pltaylor16/x-cut)
([Taylor et al., 2020](https://arxiv.org/abs/2007.00675)), [CAMB](https://camb.info/)
([Lewis et al., 1999](https://arxiv.org/abs/astro-ph/9911177)), and
[Astropy](https://www.astropy.org) 
([Robitaille et al., 2013](https://www.aanda.org/articles/aa/abs/2013/10/aa22068-13/aa22068-13.html)) 
code packages. Accordingly, if
using this module, please also cite those works.

## Pre-requisites

This module requires Python 3, Numpy, Scipy, Matplotlib, Astropy, CAMB, and its
Python wrapper.

## Installation

To use this module, simply add it to your PYTHONPATH and import the required
classes. If using Anaconda (with conda-build installed), the simplest way to achieve this is using the
command:

```bash
conda develop path/to/this/code
```

If recomputing the matter bispectrum, you may also need to recompile the
BiHalofit C file supplied with this code. To do so, you will require GCC. It
can then be compiled by navigating to the 'bihalofit' sub-directory, and using
the command:

```bash
gcc bihalofit.c -O2 -lm
```

## Examples of Usage

NOTE: Due to the sheer volume of computations required when generating C_\ells
or reduced shear corrections, those portions of the code take large
amounts of time to run. Multiprocessing is therefore strongly recommended.
Additionally, even when multiprocessed across 100 CPU threads, these operations
can take upwards of 24 hours to run.

To calculate angular power spectra or reduced shear corrections, begin by 
importin the module, and
instantiating the ```ShearObservables``` class. Angular power spectra, and their BNT transformed counterparts can then be
calculated with:

```python
import kcut_rs as krs
cs_inst = krs.ShearObservable(processors=4)
c_def, c_def_BNT = cs_inst.compute_default_c_ell_matrix()
```

Similarly, reduced shear corrections terms can be calculated using:

```python
import kcut_rs as krs
cs_inst = krs.ShearObservable(processors=4)
dc_def, dc_def_BNT = cs_inst.compute_RS_correction_matrix()
```

The Fisher matrix analysis can be carried out using the ```Fishercalc``` class.
Parameter constraints can be obtained using:

```python
import kcut_rs as krs
fishinst = krs.Fishercalc()
constraint_dict = fishinst.error_comp(cat='def')
```

Similarly, biases on a given parameter due to reduced shear are given by:

```python
import kcut_rs as krs
fishinst = krs.Fishercalc()
om_bi = fishinst.bias_calc('Om', cat='def') # Gives the bias on Omega_m
```

To carry out Fisher calculations applying the k-cut cosmic shear technique,
use the following commands:

```python
import kcut_rs as krs
fishinst = krs.Fishercalc(kcut=3.6) # k-cut given in units of Mpc^-1
constraint_dict = fishinst.error_comp(cat='BNT')
fishinst.bias_calc('Om', cat='BNT') # Gives the bias on Omega_m
```
