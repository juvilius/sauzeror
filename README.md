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

## Dependencies

+ Python
+ numpy
+ scipy
+ numba
+ matplotlib (for graphical output)
