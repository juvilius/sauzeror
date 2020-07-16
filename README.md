# SAUZEROR

## Structure Alignments Using Z-scaled EigenRanks Of Residues

```
python sauzeror.py
```

## Example

Align several domains, listed in *domains.txt*, with every structure in the *SCOPe* directory:

```
python sauzeror.py -v -mp align domains.txt ../SCOPe/
```

## Options

Run sauzeror.py without arguments to view help.

## Dependencies

+ [Python](https://docs.python.org/3/)
+ [numpy](https://numpy.org/doc/stable/)
+ [scipy](https://docs.scipy.org/doc/scipy/reference/)
+ [numba](http://numba.pydata.org/)
+ [matplotlib (for graphical output)](https://matplotlib.org/3.2.1/index.html)
+ [atomium (as a parser option)](https://github.com/samirelanduk/atomium)
