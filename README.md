# SAUZEROR

## Structure Alignments Using Z-scaled EigenRanks Of Residues

Implementation of the novel _EigenRank_ approach by [Heinke et al.](https://doi.org/10.1007/978-3-030-19093-4_18).

```
python sauzeror.py
```

## Example

Align several domains, listed in _domains.txt_, with every structure in the _SCOPe_ directory and save the output under _results.txt_:

```
python sauzeror.py -v align domains.txt ../SCOPe/ -o results.txt
```

## Options

Run sauzeror.py without arguments to view a help message.

## Tips

+ Please only use this tool in a sort of batch situation.
Importing and firing up multiprocessing can take about a second. 
numba compiles the machine code for another half second the first time the script is runs.  
No worries though: if you have some hundred or thousand structures, each alignment will take only milliseconds.

+ Use atomium if you have anything but _normal_ .pdb files (i.e. mmcif) or if you only have the PDB-IDs for it to fetch.
The primary parser works fine otherwise and is faster.  


+ Only the first chain of the first model is selected of each structure file. There are two options should you want to analyse more multiple chains per structure:

        1. split the file so that there's only one chain per file
        2. change the script to use all chains

For assistance with SAUZEROR email me at <jvoigt4@hs-mittweida.de>. I'm also open to any discussion regarding the script or it's underlying method.
Variants of SAUZEROR and other stuff is at <https://github.com/juvilius/sauzeror-etc>.

## Requirements

+ [Python](https://docs.python.org/3/)
+ [numpy](https://numpy.org/doc/stable/)
+ [scipy](https://docs.scipy.org/doc/scipy/reference/)
+ [numba](http://numba.pydata.org/)
+ [matplotlib (for graphical output)](https://matplotlib.org/3.2.1/index.html)
+ [atomium (as a parser option)](https://github.com/samirelanduk/atomium)
