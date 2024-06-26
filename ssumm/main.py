from argparser import parse_args
from SSumM import SSumM
import math

def main():
    args = parse_args()
    dataPath = args.file
    fracK = math.ceil(args.k * 100) / 100.0  # math.ceil(args.k * 100) 括号别错了

    print("---------------------------------------------------")
    if args.re == 1:
        raise NotImplementedError("ReOne Class")
        # graph = ReOne(dataPath, outputfolder, fracK, seed, use_fast_topndrop)
    else:
        graph = SSumM(dataPath, args.out, fracK, args.seed, args.use_fast_topndrop, args.urd)
    graph.inputGraph()
    elapsTime = graph.Summarize(args.T, args.cals)
    if args.sidx == 0:
        graph.summarySave_initid()
    else:
        graph.summarySave_newidx()

    originalSize = 2 * graph.numEdges * math.log2(graph.numNodes)
    summarySize = graph.numNodes * math.log2(graph.numSuperNodes) + graph.numSuperEdges * (
            2 * math.log2(graph.numSuperNodes) + math.log2(graph.maxWeight))
    print("---------------------------------------------------")
    for key, value in dict(args._get_kwargs()).items():
        print(f"{key}: {value}", end=";  ")
    print("\n---------------------------------------------------")
    print(f"Dataset Name:   {graph.dataset_name}    fracK:   {fracK}")
    print(
        f"Summary size:   {originalSize:.2f} bits  -->  {summarySize:.2f} bits ({100 * summarySize / originalSize:.6f}%)")
    print(f"|V|:  {graph.numNodes}  |E|:  {graph.numEdges}  |S|:  {graph.numSuperNodes}  |P|:  {graph.numSuperEdges}")
    print(f"Elapsed Time:   {elapsTime} s")
    REDSTR = ['\033[91m', '\033[0m']
    print(f"L{args.re}error:   {REDSTR[0]}{graph.norm()[1]:.4e}{REDSTR[1]}")  # flag main

    # if graph.debug:
    #     print(graph.removed_node)


if __name__ == "__main__":
    main()