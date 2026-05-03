# Paper 2 §10 — semantic selectivity analysis of PoE-2 ratios

Class assignment of each expert:

| expert | class |
|---|---|
| long | style |
| formal | style |
| positive | sentiment |
| positive2 | sentiment |
| concrete | topic |
| sports | topic |

## Per-pair table (sorted by ratio)

| pair | class composition | ratio | regime |
|---|---|---:|---|
| formal × sports | style×topic | 0.111 | STRONG sub-add |
| formal × concrete | style×topic | 0.400 | STRONG sub-add |
| long × positive | sentiment×style | 0.533 | moderate sub-add |
| long × concrete | style×topic | 0.567 | moderate sub-add |
| formal × positive2 | sentiment×style | 0.611 | moderate sub-add |
| long × formal | same | 0.661 | moderate sub-add |
| long × positive2 | sentiment×style | 0.713 | moderate sub-add |
| long × sports | style×topic | 0.759 | moderate sub-add |
| positive × concrete | sentiment×topic | 0.760 | moderate sub-add |
| formal × positive | sentiment×style | 0.795 | moderate sub-add |
| concrete × sports | same | 0.937 | moderate sub-add |
| positive2 × concrete | sentiment×topic | 1.049 | super-add |
| positive2 × sports | sentiment×topic | 1.189 | super-add |
| positive × sports | sentiment×topic | 1.189 | super-add |
| positive × positive2 | same | 1.681 | super-add |

## Class-level aggregation

| class | n | mean | median | min | max | super-add count |
|---|---:|---:|---:|---:|---:|---:|
| same | 3 | 1.093 | 0.937 | 0.661 | 1.681 | 1/3 |
| sentiment×topic | 4 | 1.047 | 1.119 | 0.760 | 1.189 | 3/4 |
| sentiment×style | 4 | 0.663 | 0.662 | 0.533 | 0.795 | 0/4 |
| style×topic | 4 | 0.459 | 0.484 | 0.111 | 0.759 | 0/4 |

## Bigger split: style-containing vs no-style pairs

| split | n | mean | super-add count |
|---|---:|---:|---:|
| any-style (formal or long) | 9 | 0.572 | 0/9 |
| no-style | 6 | 1.134 | 4/6 |

**Reading**: the super-additive pairs cluster among the no-style ones; all 9 style-containing pairs are sub-additive. The split corresponds to a 2× difference in mean ratio (~1.13 vs ~0.57).
