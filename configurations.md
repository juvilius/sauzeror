# SAUZEROR configurations

## Flavour of SAUZEROR as file prefix

| number | what’s different     |
| :----: | :------------------- |
|   \_   | normal branch        |
|   3    | alignment refinement |
|   4    | more PCs             |
|   6    | LRs instead          |
|   7    | 3+4                  |
|   9    | 3+6                  |

## Distance matrix

  - euclidean –\> `scipy.spatial.distance_matrix`

  - normal distribution, CDF –\> `scipy.special.ndtr`

## ER/LR profiles

  - different amounts of principal components –\> change indices of
    `princo`

  - z-scaling of LRs before/after –\> `scale2` (from R, different to
    `scipy.stats.scale`)

## Alignment/SW algorithm

  - `gap` and `limit` parameters  
    –\> `[0-9]npanalysis.py` for analysis of optimal new parameters

  - refinement step

## Parameters for different profiles

|                  description                  | gap | limit |
| :-------------------------------------------- | --- | ----- |
| *normal* ER, 1st PC of scaled or unscaled LRs | 0.8 | 0.7   |
|                                               | 0.5 | 1.4   |
| ER with 1st **and** 2nd PC                    | 0.8 | 2.0   |
| scaled LR                                     | 2.2 | 4.2   |
| refinement step                               | 0.4 | 0.2   |
