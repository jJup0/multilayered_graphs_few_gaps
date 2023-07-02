# Notes for Code for Bachelor Thesis: Multilayered Graphs with Minimal Crossings and Few Gaps

## pygraphviz install

**Exact command used to install pygraphviz:**

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

- [x] implement existing algorithm with constrained nodes: split into two gaps first and apply heuristic
- [ ] make one-sided crossing minimization of algorithms
- [ ] performance comparison graphs (as in lines on x,y plane showing size vs crossings and size vs time)
- [ ] two-sided ILP formulation
      https://www.researchgate.net/profile/Brian-Alspach/publication/221335384_Arc_Searching_Digraphs_Without_Jumping/links/00b7d525cb7012e03d000000/Arc-Searching-Digraphs-Without-Jumping.pdf#page=273
      page 276
- [ ] different random graph types:
  - [ ] networkx random cluster graphs https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_clustered.random_clustered_graph.html#networkx.generators.random_clustered.random_clustered_graph
  - [ ] networkx other random graphs: https://networkx.org/documentation/stable/reference/generators.html#module-networkx.generators.random_graphs
  - [ ] networkx real graphs: https://networkx.org/documentation/stable/reference/generators.html#module-networkx.generators.social
- [ ] analyse 1- and 2-layer crossing minimization for graph with only 2 layers
- [ ] Bleibt 3-approximation für median heuristic
- [ ] multiple gaps
- [ ] ILP max size realistic computation time
- [ ] ILP multiple gaps (more than 2 choices)
- [ ] Abstract for presentation
- [ ] presentation: definitions, related work, work done so far, future plans
- [ ] gurobi set gap noe order fixed (order by incoming edge)
- [ ] where to place k gaps (dp approach dp[i][j][k] = upto ith gap node, starting at jth real node, using k gaps)
