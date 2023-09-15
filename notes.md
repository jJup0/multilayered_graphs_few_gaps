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

- [x] k gaps dp algorithm: had extra unnecessary loop over prev_gap_idx, now only looping over prev_vnode_idx
- [x] gurobi k gaps but only for one-sided (variable for gap between nodes, sum must not exceed k)
- Median side gaps: virtual node will have identical crossing count in gurobi and heuristic layout so approximation remains
- [x] start writing

---

- [x] Gurobi with integer variables:
      modeled with boolean variables instead: not boolean relative ordering variable but boolean to signify whether a node is at position or not
      did not fully implement because crossing objective requires helper variables of relative ordering
- [x] gurobi set gap node order fixed (order by incoming edge)
- [x] proof for k gaps 3-heuristic
- [x] Bleibt 3-approximation für median heuristic

---

- [x] side gap heuristic: binary search for splitting point
- [x] Proof gap nodes do not cross
- [ ] ILP max size realistic computation time

k-gaps proof:

- starting from optimal
- merge gap nodes into a single node
- apply median heuristic
- split merged gap nodes up again
- this ordering will be found by median + kgaps dp

Approximation ratio/factor
can define function in preliminaries e.g. get virtual nodes in order of neighbor
one/two sentence summary oflonger proofs if simple explanation is possible

- [ ] no need for explicitly explaining binary search and having it in the pseudo code, just state that partition index needs to be found and this can be done with binary search
      "our algorithm computes a 3-approximation to the problem <give probem a name like SG-OSCM (side gap one sided crossing minimization)>"
      not clear what c\_{opt} i referring to exactly (optimal crossings for real nodes) [on page 11]
      where to fit in two layer ILP formulations (does not fit into one sided thoery chapters)
      give exact big-O notation runtimes for algorithms

look for a case study example graph in existing papers

Chapter: Experimention

- implementierung
- quantitative evalutation
- case study

Meeting 24-08

- layered graph drawing a pi for a layered graph G
- gap is maximal sequence of consecutive nodes s.t...
- "preliminary experiments show that one gap creates unreadable graph"
- this heuristic may have an approximation ratio
- \textrm may be not cursive, check how it compares to \text
