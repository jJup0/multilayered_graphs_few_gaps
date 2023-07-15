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

- [x] implement existing algorithm with constrained nodes: split into two gaps first and apply heuristic (broken)
- [x] check if constraints variation is still broken (it is)
- [x] make one-sided crossing minimization of algorithms
- [x] two-sided ILP formulation
      https://www.researchgate.net/profile/Brian-Alspach/publication/221335384_Arc_Searching_Digraphs_Without_Jumping/links/00b7d525cb7012e03d000000/Arc-Searching-Digraphs-Without-Jumping.pdf#page=273
      page 276
- [x] analyse 1- and 2-layer crossing minimization for graph with only 2 layers
- [x] performance comparison graphs (as in lines on x,y plane showing size vs crossings and size vs time)
- [x] networkx different random graph types (turns out they are all not really useful, should just generate on own):
  - [x] networkx random cluster graphs https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_clustered.random_clustered_graph.html#networkx.generators.random_clustered.random_clustered_graph
  - [x] networkx other random graphs: https://networkx.org/documentation/stable/reference/generators.html#module-networkx.generators.random_graphs
  - [x] networkx real graphs: (don't seem to be too useful for testing) https://networkx.org/documentation/stable/reference/generators.html#module-networkx.generators.social
- [x] multiple gaps
- [x] where to place k gaps (dp approach dp[i][j][k] = upto ith gap node, starting at jth real node, using k gaps)

---

- [ ] gurobi k gaps but only for one-sided (variable for gap between nodes, sum must not exceed k)
- [ ] proof for k gaps 3-heuristic
- [ ] start writing
- [ ] Bleibt 3-approximation für median heuristic
- [ ] ILP max size realistic computation time
- [ ] gurobi set gap node order fixed (order by incoming edge)
