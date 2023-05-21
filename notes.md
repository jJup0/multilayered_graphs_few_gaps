# Notes for Code for Bachelor Thesis: Multilayered Graphs with Minimal Crossings and Few Gaps

## pygraphviz install

__Exact command used to install pygraphviz:__

python -m pip install --global-option=build_ext --global-option="-IC:\programming_prog_files\Graphviz\include" --global-option="-LC:\programming_prog_files\Graphviz\lib" pygraphviz

note: replace "\programming_prog_files\Graphviz" with path to graphviz


## Possible libraries to use

out of the box, straight edges, virtual nodes are shown

https://github.com/gml4gtk/pysugiyama

more developed sugiyama, straight edges, cannot give constraints/orders

https://pypi.org/project/grandalf/

 more developed sugiyama, straight edges ...

https://pypi.org/project/ogdf-python/




alternative corrsing minimization strategy: find relative order of gap nodes, and find index where left/right split results in fewest crossings 

larger instances, automated crossing calculation

networkx ranodm cluster graphs

in tunet vpn sein um key zu installieren

implemtation of constrained graph - split into two gaps first and apply heuristic