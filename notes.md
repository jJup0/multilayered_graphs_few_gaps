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


## TODOS

- networkx random cluster graphs https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_clustered.random_clustered_graph.html#networkx.generators.random_clustered.random_clustered_graph
  - random graphs: https://networkx.org/documentation/stable/reference/generators.html#module-networkx.generators.random_graphs
  - real graphs: https://networkx.org/documentation/stable/reference/generators.html#module-networkx.generators.social
- implement existing algorithm with constrained nodes: split into two gaps first and apply heuristic
- two-sided ILP program