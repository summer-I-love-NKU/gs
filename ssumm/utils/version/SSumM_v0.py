import argparse
import os
from datetime import datetime
import numpy as np
import random
from collections import defaultdict
import math


# v0 python版本的SSumM，已经检查无误


class MyRandom:
    def __init__(self):
        self.seed

    def seed(self, _seed):
        self.seed = _seed % 2147483647  # 确保种子在一个合适的范围内

    def generate(self):
        # 线性同余算法参数（这些参数是常用的，但不是唯一的）
        a = 16807
        m = 2147483647  # 模数，一个大的质数

        self.seed = (a * self.seed) % m
        return self.seed

    # 生成一个在[low, high]之间的随机整数
    def randint(self, low, high):
        return low + self.generate() % (high - low + 1)


class SSumM:
    def __init__(self, _dataPath: str, _outputfolder: str, _fracK: float, _random_seed: int, _use_fast_topndrop: int,
                 urd: int, useNE: int):

        self.debug = 0  # flag debug
        self.removed_node = []

        self.numNodes = 0
        self.numEdges = 0
        self.numSuperNodes = 0
        self.numSuperEdges = 0
        self.targetSummarySize = 0
        self.node2Idx = defaultdict(int)
        self.deg = []
        self.edges = None
        self.num4precision = 1e4

        self.dataPath = _dataPath
        self.dataset_name = _dataPath.split('/')[-1].split('\\')[-1][:-4]  # xxx.txt
        self.outputfolder = _outputfolder
        self.fracK = _fracK
        self.random_seed = _random_seed
        self.use_fast_topndrop = _use_fast_topndrop

        self.divThreshold = 500
        self.pair_used = np.zeros((self.divThreshold, self.divThreshold))
        self.cntFlag = 0
        self.isolatedId = -1
        self.superGraph = None
        self.insideSupernode = defaultdict(list)
        self.snList = []
        self.rep = []
        self.dataCost = []
        self.modelCost = []
        self.maxWeight = 1

        self.random = MyRandom() if urd else random
        self.random.seed(self.random_seed)

    def deprint(self, *args, sep=' ', end='\n', file=None):
        if self.debug:
            print(*args, sep=' ', end='\n', file=None)

    def Summarize(self, T, k):
        print("Start Time: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        start_time = datetime.now()

        for m in range(1, T + 1):  # flag summarize
            print("iter:", m)
            if m != T:
                threshold = 1.0 / (1 + m)
                threshold = int(threshold * 1e6) / 1e6
            else:
                threshold = 0

            sorted_shingle_list = self.signature_seed(k) if self.random_seed != -1 else self.signature(k)

            similar_node_set = self.divSupernode(sorted_shingle_list, k)

            check = self.mergeStep(similar_node_set, threshold)

            if check == 0:
                break
        # summarySize = self.numNodes * math.log2(self.numSuperNodes) + self.numSuperEdges * (2 * math.log2(self.numSuperNodes) + math.log2(self.maxWeight));print(summarySize / self.targetSummarySize, summarySize / self.targetSummarySize * self.fracK)
        self.dropEdgesAndmergeIsolated()
        # summarySize = self.numNodes * math.log2(self.numSuperNodes) + self.numSuperEdges * (2 * math.log2(self.numSuperNodes) + math.log2(self.maxWeight));print(summarySize / self.targetSummarySize, summarySize / self.targetSummarySize * self.fracK)
        if self.targetSummarySize < self.numNodes * math.log2(self.numSuperNodes) + self.numSuperEdges * (
                2 * math.log2(self.numSuperNodes) + math.log2(self.maxWeight)):
            if self.use_fast_topndrop == 0:
                self.topNDrop_sort()
            else:
                self.topNDrop()
        end_time = datetime.now()
        return (end_time - start_time).seconds

    def mergeStep(self, similar_node_set, threshold):
        superedgeCost = 2 * math.log2(self.numSuperNodes) + math.log2(self.maxWeight)
        for _i, tmp in enumerate(similar_node_set):
            ###############################
            # 错误代码案例！！！
            # if len(tmp) < 2:
            #     continue
            #
            # numskip_patience = 0
            # while numskip_patience < max(math.log2(len(tmp)), 1):
            #     length = len(tmp)
            ###############################

            # 正确的是这样：length是在开始定义的！！！  _attn 2023.1.3 第0个bug（和第2个bug是一处区域）
            # 而且后面：max(math.log2(length), 1)必须用length而不是len(tmp)
            length = len(tmp)
            if length < 2:
                continue

            numskip_patience = 0  # flag mergeStep
            while numskip_patience < max(math.log2(length),
                                         1):  # _attn 2024.1.4发现第2个bug，这里也要改成length，必须用length而不是len(tmp)！因为tmp不会真的删除元素，只是把length缩减了！！！#while numskip_patience < max(math.log2(len(tmp)), 1):

                if length < 2:
                    break

                self.cntFlag += 1
                uniqueValue = self.cntFlag
                bestSrc, bestDst, maxSaving = -1, -1, -1

                for i in range(length):

                    #  python的random的randint(0, n)是[0, n] java的Random的nextInt(n)是[0,n)
                    rn = self.random.randint(0, length * (length - 1) - 1)
                    src = rn // (length - 1)
                    dst = rn % (length - 1)
                    if src <= dst:
                        dst += 1

                    if self.pair_used[src][dst] == uniqueValue:
                        continue
                    else:
                        self.pair_used[src][dst] = uniqueValue
                    saving = self.savingMDL(tmp[src], tmp[dst], superedgeCost)  # flag savingMDL
                    # self.deprint(saving)
                    if saving > maxSaving:
                        maxSaving = saving
                        bestSrc = src
                        bestDst = dst

                if maxSaving < threshold:
                    numskip_patience += 1
                    continue

                numskip_patience = 0
                self.merge(tmp[bestSrc], tmp[bestDst])

                length -= 1
                tmp[bestDst] = tmp[length]

                # Update variables
                superedgeCost = 2 * math.log2(self.numSuperNodes) + math.log2(self.maxWeight)
                summarySize = self.numNodes * math.log2(self.numSuperNodes) + self.numSuperEdges * (
                            2 * math.log2(self.numSuperNodes) + math.log2(self.maxWeight))
                summarySize = self.numNodes * math.log2(self.numSuperNodes) + self.numSuperEdges * (
                        2 * math.log2(self.numSuperNodes) + math.log2(self.maxWeight))
                # print("###",summarySize / self.targetSummarySize, summarySize / self.targetSummarySize * self.fracK)
                # 当 summarySize <= self.targetSummarySize 时候结束，但最后summarySize会比实际要求的小一些，因为需要删除虚拟边，合并孤立节点
                if summarySize <= self.targetSummarySize:
                    return 0
        return 1  # Not successful, needs to continue merging

    def norm(self):
        su = [0] * self.numNodes
        for v in self.snList:
            for _u in self.insideSupernode[v]:
                su[_u] = v

        err = 0
        for edge in self.edges:
            if edge < 0:
                continue
            _u = edge >> 32
            _v = edge & 0x7FFF_FFFF
            if _u == _v:
                continue

            v = su[_v]
            u = su[_u]
            edgeCnt = self.superGraph[u].get(v, 0)
            sz = len(self.insideSupernode[u])
            sz *= (len(self.insideSupernode[u]) - 1) if u == v else len(self.insideSupernode[v])
            w = edgeCnt / sz
            err += ((1 - w) * (1 - w) - w * w)

        err *= 2
        for v in self.snList:
            for u in self.superGraph[v]:
                sz = len(self.insideSupernode[u])
                sz *= len(self.insideSupernode[u]) if u == v else len(self.insideSupernode[v])
                edgeCnt = self.superGraph[v][u]
                w = edgeCnt / sz
                err += (w * w * sz)

        err = np.sqrt(err)
        return err, err / self.numNodes / (self.numNodes - 1)

    ''' -------------------------- 基本函数 ---------------------------'''

    # def log2(self,x):
    #     return math.log(x) / self.log2num

    def CutPrecision(self, x):
        return ((int)(x * self.num4precision)) / self.num4precision

    def addNode(self, v):
        if v not in self.node2Idx:
            self.node2Idx[v] = self.numNodes
            idx = self.numNodes
            self.deg.append(0)
            self.numNodes += 1
        else:
            idx = self.node2Idx[v]
        self.deg[idx] = self.deg[idx] + 1

    def inputGraph(self):
        print("Data Read Start: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # 得到 numNodes
        cnt = 0
        with open(self.dataPath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                src, dst = map(int, parts[:2])
                self.addNode(src)
                self.addNode(dst)
                cnt += 1
        # 得到 numEdges
        self.superGraph = [{} for _ in range(self.numNodes)]
        self.edges = [0] * cnt
        with open(self.dataPath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                src, dst = map(int, parts[:2])
                src, dst = self.node2Idx[src], self.node2Idx[dst]
                self.superGraph[src][dst] = 1
                self.superGraph[dst][src] = 1
                self.edges[self.numEdges] = (src << 32) + dst
                self.numEdges += 1

        print("|V|: ", self.numNodes, "    |E|: ", self.numEdges)
        # 超级节点相关
        self.insideSupernode = [[i] for i in range(self.numNodes)]
        self.rep = list(range(self.numNodes))
        self.snList = list(range(self.numNodes))

        self.numSuperNodes = self.numNodes
        self.numSuperEdges = self.numEdges

        self.maxWeight = 1
        self.dataCost = [0.0] * self.numNodes
        self.modelCost = list(map(len, self.superGraph))

        self.targetSummarySize = self.fracK * 2 * self.numEdges * math.log2(self.numNodes)
        print("Finished reading the input graph")

    def summarySave_initid(self):
        idx2Node = {v: k for k, v in self.node2Idx.items()}

        output_folder = self.outputfolder + self.dataset_name + '/'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        filename = f"{self.dataset_name}_{self.fracK}.snlist"
        with open(output_folder + filename, 'w') as f:
            for sup_v in self.snList:
                line = str(idx2Node[sup_v]) + '\t' + '\t'.join(
                    str(idx2Node[sub_v]) for sub_v in self.insideSupernode[sup_v])
                print(line, file=f)

        filename = f"{self.dataset_name}_{self.fracK}.suedge"
        with open(output_folder + filename, 'w') as f:
            for sup_v in self.snList:
                for neighbor_v in self.superGraph[sup_v].keys():
                    if sup_v >= neighbor_v:
                        weight = self.superGraph[sup_v][neighbor_v]
                        # Why are there many? sup_v == neighbor_v
                        line_weight = weight // 2 if sup_v == neighbor_v else weight
                        line = f"{idx2Node[sup_v]}\t{idx2Node[neighbor_v]}\t{line_weight}"
                        print(line, file=f)

    def summarySave_newidx(self):
        raise NotImplementedError("summarySave_newidx")

    '''--------------------节点相似性计算和合并节点的函数 ---------------------'''

    def getRandomPermutation(self, length):
        # 用randint 而不是 shuffle ，因为可以用种子控制
        array = list(range(length))
        for i in range(length):
            ran = i + self.random.randint(0, length - i - 1)  # python的random.randint和java的random.nextInt不一样！
            array[i], array[ran] = array[ran], array[i]
        return array

    def signature_seed(self, k):
        res_ans = np.zeros((self.numSuperNodes, k + 1), dtype=int)

        for res_i in range(k):
            ans = np.zeros(self.numNodes, dtype=int)
            superAns = np.zeros(self.numSuperNodes, dtype=int)
            f_hash_value = self.getRandomPermutation(self.numNodes)

            ans[:] = f_hash_value  # ans=f_hash_value的话ans就成为list了
            for e in self.edges:
                src = e >> 32
                dst = e & 0x7FFFFFFF
                ans[src] = min(f_hash_value[dst], ans[src])
                ans[dst] = min(f_hash_value[src], ans[dst])

            superAns[:] = 0x7FFFFFFF
            for i in range(self.numSuperNodes):
                # // snList 存储了超级节点的原始id！！！
                supernode = self.snList[i]
                superAns[i] = min([ans[j] for j in self.insideSupernode[supernode]])

            res_ans[:, res_i] = superAns  # 对！
        res_ans[:, k] = np.array(self.snList)

        # 错误的：但这是啥意思 sorted_array = res_ans[res_ans[:, :k].argsort(kind='mergesort')]
        # 正确的：
        # array([[10., 3., 2.],
        #        [10., 4., 3.],
        #        [0., 5., 1.],
        #        [20., 5., 4.],
        #        [0., 4., 0.]])
        # array([[0., 4., 0.],
        #        [0., 5., 1.],
        #        [10., 3., 2.],
        #        [10., 4., 3.],
        #        [20., 5., 4.]])
        sorted_array = np.array(sorted(res_ans, key=lambda x: tuple(x)))
        return sorted_array

    def signature(self, k):
        raise NotImplementedError("signature with multi thread")

    def listN(self, array, first):
        ans = []
        tmp = array[0]
        last = first - 1

        for i in range(len(array)):
            if tmp == array[i]:
                last += 1
            else:
                model = [first, last]
                ans.append(model)
                tmp = array[i]
                first = last + 1
                last += 1

        model = [first, last]
        ans.append(model)

        return ans

    def get_column_element(self, array, col_index, first, last):
        ans = [array[i][col_index] for i in range(first, last + 1)]
        return ans

    def divSupernode(self, sorted_shingle_list, k):
        shingle_value = [sorted_shingle_list[i][0] for i in range(self.numSuperNodes)]
        continue_samenum_idx = self.listN(shingle_value, 0)
        return self.recursiveDiv(sorted_shingle_list, continue_samenum_idx, 0, k)

    def recursiveDiv(self, sorted_shingle_list, continue_samenum_idx, col, k):
        ans = []
        for x in continue_samenum_idx:
            first = x[0]
            last = x[1]
            length = last - first + 1
            if length <= self.divThreshold:
                ans.append(self.get_column_element(sorted_shingle_list, k, first, last))
            else:
                if col < k:
                    col += 1
                    t_next_shingle = self.get_column_element(sorted_shingle_list, col, first, last)
                    new_continue_samenum_idx = self.listN(t_next_shingle, first)
                    ans.extend(self.recursiveDiv(sorted_shingle_list, new_continue_samenum_idx, col, k))
                    col -= 1
                elif col == k:
                    while first + self.divThreshold - 1 < last:
                        ans.append(
                            self.get_column_element(sorted_shingle_list, k, first, first + self.divThreshold - 1))
                        first += self.divThreshold
                    ans.append(self.get_column_element(sorted_shingle_list, k, first, first + self.divThreshold - 1))
        return ans

    ''' ---------------------------- 代价计算函数 ---------------------------'''

    def entropy(self, p):
        if (p == 0.0) or (p == 1.0):
            entropy = 0
        else:
            entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        return entropy

    def savingMDL(self, src, dst, edgeCost):
        tmpModelCost = edgeCost * (self.modelCost[src] + self.modelCost[dst])
        tmpDataCost = self.dataCost[src] + self.dataCost[dst]
        # 善用 dict 的 get， 和 java的getOrdefault一样，java的get找不到是0，而python是None
        tmpModelCost -= edgeCost if self.superGraph[src].get(dst, 0) > 0 else 0
        tmpDataCost -= self.dataCostPair(src, dst)
        denominator = tmpModelCost + tmpDataCost

        merged_nodeset = self.mergeNodeSet(src, dst)

        tmpMergeCost = 0
        for n in merged_nodeset:
            tmpData = self.TotalCostPair_merge(src, dst, n)
            tmpMergeCost += -tmpData if tmpData < 0 else tmpData
        # if denominator==0:??? 不会的！
        res = 1 - (tmpMergeCost / denominator)
        res = self.CutPrecision(res)  # _attn 控制位数 只在这里用一次 2024.1.5凌晨 第5个bug，其实不算bug，是精度问题
        return res

    def mergeNodeSet(self, src, dst):
        returnSet = set(self.superGraph[src].keys()) | set(self.superGraph[dst].keys())
        if dst in returnSet:
            returnSet.add(src)
            returnSet.remove(dst)
        return sorted(returnSet)  # 有序

    def dataCostPair(self, src, dst):

        tmpEdgeNum = self.superGraph[src].get(dst, 0) & 0x7FFFFFFF  # 这里很巧妙！！！

        if self.superGraph[src].get(dst,
                                    0) > 0:  # if tmpEdgeNum > 0:#_attn 2024.1.4发现第3个bug：这里应该是 if(superGraph[src].get(dst,0) > 0)判断是否是sparse情况,因为这个值可能为负数！ 不能用tmpEdgeNum因为tmpEdgeNum一定是>=0的
            srcSize = len(self.insideSupernode[src])
            matrixSize = srcSize * (srcSize - 1 if src == dst else len(self.insideSupernode[dst]))
            # weight = tmpEdgeNum if matrixSize==0 else  tmpEdgeNum / matrixSize #  if matrixSize==0 ？不会的！
            weight = tmpEdgeNum / matrixSize
            tmpDataCost = self.entropy(weight) * matrixSize
        else:
            tmpDataCost = 2 * math.log2(self.numNodes) * tmpEdgeNum
        if src == dst:
            tmpDataCost /= 2
        return tmpDataCost

    def TotalCostPair_merge(self, src, dst, n):
        source = self.superGraph[src]
        dest = self.superGraph[dst]

        matrixSize = len(self.insideSupernode[src]) + len(self.insideSupernode[dst])
        # 1 & 0x7FFFFFFF + 1
        # Out[134]: 0
        # (1 & 0x7FFFFFFF) + 1
        # Out[135]: 2
        if n == src:
            tmpEdgeNum = ((source.get(src, 0) & 0x7FFF_FFFF) + 2 * (source.get(dst, 0) & 0x7FFF_FFFF) + (
                        dest.get(dst, 0) & 0x7FFF_FFFF)) // 2
            matrixSize = matrixSize * (matrixSize - 1) // 2
        else:
            tmpEdgeNum = (source.get(n, 0) & 0x7FFF_FFFF) + (dest.get(n, 0) & 0x7FFF_FFFF)
            matrixSize *= len(self.insideSupernode[n])

        weight = tmpEdgeNum / matrixSize

        # dense = 2 * math.log2(self.numNodes) + math.log2(self.maxWeight) + self.entropy(weight) * matrixSize#_attn 2023.1.4第1个bug（哦是2024年了）逐行debug比对java和python结果差异，发现了这个bug:numNodes应该改成numSuperNodes，python代码是用gpt4生成之后再修改的，检查的不仔细啊！感谢自己的完美主义、严谨、精益求精、刨根问底、求是精神，慢就是快！
        dense = 2 * math.log2(self.numSuperNodes) + math.log2(self.maxWeight) + self.entropy(weight) * matrixSize

        sparse = 2 * math.log2(self.numNodes) * tmpEdgeNum
        tmpMergeCost = dense if dense <= sparse else -sparse

        return tmpMergeCost

    '''----------------------- 合并相关操作函数 ----------------------------'''

    def merge(self, src, dst):
        tmpMax = 0
        source = self.superGraph[src]
        dest = self.superGraph[dst]

        merged_NodeSet = self.mergeNodeSet(src, dst)

        for n in merged_NodeSet:
            denseNum = 0
            if n == src:
                tmpEdgeNum = (source.get(src, 0) & 0x7FFF_FFFF) + 2 * (source.get(dst, 0) & 0x7FFF_FFFF) + (
                            dest.get(dst, 0) & 0x7FFF_FFFF)
                if source.get(src, 0) > 0:
                    denseNum += 1
                if source.get(dst, 0) > 0:
                    denseNum += 1
                if dest.get(dst,
                            0) > 0:  # _attn 2023.1.4 第4个bug 看到删除159个节点后，numSuperEdges不一样！导致summarySize不一样，而涉及到它的写入，一处是numSuperEdges -= (denseNum - 1);另一处是dropEdgesAndmergeIsolated那里，猜测 大概率与 denseNum有关，终于发现这里的粗心，是dest.get(dst,0)而不是dest.get(src,0)！！！
                    denseNum += 1
            else:
                tmpEdgeNum = (source.get(n, 0) & 0x7FFF_FFFF) + (dest.get(n, 0) & 0x7FFF_FFFF)
            tmpMax = max(tmpMax, tmpEdgeNum)

            if n != src:
                dataTmp = self.dataCost[n] - (self.dataCostPair(src, n) + self.dataCostPair(dst, n))
                modelTmp = self.modelCost[n]
                if self.superGraph[src].get(n, 0) > 0:
                    modelTmp -= 1
                    denseNum += 1
                if self.superGraph[dst].get(n, 0) > 0:
                    modelTmp -= 1
                    denseNum += 1

                self.dataCost[n] = dataTmp
                self.modelCost[n] = modelTmp

            tmpMergeCost = self.TotalCostPair_merge(src, dst, n)
            if tmpMergeCost >= 0:
                self.numSuperEdges -= (denseNum - 1)
                self.modelCost[n] += 1
            else:
                self.numSuperEdges -= denseNum
                # // Java里：按位或， 变成负数了！！！！
                # tmpEdgeNum |= 0x80000000;
                # python按位或需要这样
                tmpEdgeNum = (tmpEdgeNum | 0x8000_0000) - 0x1_0000_0000

            # Update src tmpEdgeNum
            self.superGraph[src][n] = tmpEdgeNum
            self.superGraph[n][src] = tmpEdgeNum
            # java这里是superGraph[n].remove(dst) 返回值是value（存在）或者0（不存在）， n有可能是src的邻居而不是dst的邻居，所以superGraph[n][dst]可能不存在！
            #  需要先判断，而不是del self.superGraph[n][dst]
            if dst in self.superGraph[n]:
                del self.superGraph[n][dst]

        if tmpMax > self.maxWeight:
            self.maxWeight = tmpMax

        self.superGraph[dst] = None  # del self.superGraph[dst]#不要del 要保证superGraph维度不变！！！
        self.insideSupernode[src].extend(self.insideSupernode[dst])  # 列表extend！
        self.insideSupernode[dst] = []  # 这里不要 del self.insideSupernode[dst] 同样是为了保证insideSupernode维度不变

        merged_NodeSet.append(src)
        merged_NodeSet.sort()
        for n in merged_NodeSet:
            if n == src:
                tmpDataCost = 0
                tmpModelCost = 0
                for m in self.superGraph[src].keys():
                    tmpDataCost += self.dataCostPair(src, m)
                    tmpModelCost += 1 if self.superGraph[src].get(m, 0) > 0 else 0
                self.dataCost[src] = tmpDataCost
                self.modelCost[src] = tmpModelCost
            else:
                self.dataCost[n] += self.dataCostPair(src, n)
        self.dataCost[dst] = 0
        self.modelCost[dst] = 0
        self.removeSuperNode(dst)

    def removeSuperNode(self, target):
        # self.deprint('remove',target)
        # if self.debug:
        #     self.removed_node.append(target)

        self.numSuperNodes -= 1
        last_node = self.snList[self.numSuperNodes]
        delete_node_idx = self.rep[target]

        self.snList[delete_node_idx] = last_node
        self.rep[last_node] = self.rep[target]
        self.rep[target] = -1
        self.snList.pop()

    def dropEdgesAndmergeIsolated(self):

        # for (int n: snList){
        #     superGraph[n].int2IntEntrySet().removeIf(e -> (e.getIntValue() < 0));
        # }
        for n in self.snList:
            self.superGraph[n] = {k: v for k, v in self.superGraph[n].items() if v >= 0}

        i = 0
        while i < len(self.snList):
            target = self.snList[i]
            i += 1
            if len(self.superGraph[target]) == 0:  # //if(superGraph[target].size() == 0) {
                if self.isolatedId == -1:
                    self.isolatedId = target
                else:
                    self.insideSupernode[self.isolatedId].extend(self.insideSupernode[target])
                    self.superGraph[target] = None
                    self.insideSupernode[target] = []
                    self.removeSuperNode(target)
                    i -= 1

    def topNDrop(self):
        raise NotImplementedError("topNDrop")

    def topNDrop_sort(self):
        print("##### start topndrop_sort #####")
        numE = nSize = blockSize = 0
        cost = [0] * self.numSuperEdges
        sedges = [[0, 0] for _ in range(self.numSuperEdges)]
        col = 0

        for n in self.snList:
            for e in self.superGraph[n].items():
                if e[0] >= n:
                    numE = e[1] // (2 if e[0] == n else 1)
                    nSize = len(self.insideSupernode[n])
                    blockSize = nSize * ((nSize - 1) if (e[0] == n) else len(self.insideSupernode[e[0]]))
                    cost[col] = numE * numE / blockSize
                    sedges[col][0] = n
                    sedges[col][1] = e[0]
                    col += 1

        summarySize = self.numNodes * math.log2(self.numSuperNodes) + self.numSuperEdges * (
                2 * math.log2(self.numSuperNodes) + math.log2(self.maxWeight))

        while summarySize > self.targetSummarySize:
            edgeLft = self.numSuperEdges + int(
                (self.numNodes * math.log2(self.numSuperNodes) - self.targetSummarySize) / (
                        2 * math.log2(self.numSuperNodes) + math.log2(self.maxWeight)))

            indexes = sorted(list(range(len(cost))), key=lambda k: cost[k])

            for i in range(edgeLft):
                idx = indexes[i]
                src, dst = sedges[idx]
                # 这里是肯定存在的边，所以不用判断
                del self.superGraph[src][dst]
                del self.superGraph[dst][src]
                self.numSuperEdges -= 1

                if not self.superGraph[src]:  # not {} 是返回True的
                    self.mergeAndRemove(src)

                if src != dst and not self.superGraph[dst]:
                    self.mergeAndRemove(dst)

                summarySize = self.numNodes * math.log2(self.numSuperNodes) + self.numSuperEdges * (
                        2 * math.log2(self.numSuperNodes) + math.log2(self.maxWeight))

                if summarySize <= self.targetSummarySize:
                    break

    def mergeAndRemove(self, nodeId):
        if self.isolatedId == -1 or self.isolatedId == nodeId:
            self.isolatedId = nodeId
        else:
            self.insideSupernode[self.isolatedId].extend(self.insideSupernode[nodeId])
            self.superGraph[nodeId] = None
            self.insideSupernode[nodeId] = []
            self.removeSuperNode(nodeId)


def parse_args():  # flag args
    parser = argparse.ArgumentParser()

    general = parser.add_argument_group(title='general')
    general.add_argument('--file', type=str, default='', help='input file')
    general.add_argument('--out', type=str, default='./output/', help='output folder')
    general.add_argument('--k', type=float, default=0.8, help='fracK')
    general.add_argument('--re', type=int, default=2, help='compute reconstruction error, RE1 or RE2 ?')
    general.add_argument('--seed', type=int, default=2023, help='random seed')
    general.add_argument('--use_fast_topndrop', type=int, default=0,
                         help='use_fast_topndrop with median search (use_fast_topndrop=1) or just sort ('
                              'use_fast_topndrop=0) ?')
    general.add_argument('--sidx', type=int, default=0,
                         help='save result with new index (sidx=1) or initial node id (sidx=0) ?')
    general.add_argument('--T', type=int, default=20,
                         help='number of iteration')
    general.add_argument('--cals', type=int, default=10,
                         help='How many times do we calculate shingle?')
    general.add_argument('--urd', type=int, default=0,
                         help='user defined random function')
    general.add_argument('--tf', type=str, default="", help='test:input file')
    general.add_argument('--tk', type=float, default=2.0, help='test:fracK')

    args, _ = parser.parse_known_args()
    if args.tf != "":
        args.file = args.tf
    if args.tk != 2.0:
        args.k = args.tk
    return args


def main():
    args = parse_args()
    dataPath = args.file
    fracK = math.ceil(args.k * 100) / 100.0  # math.ceil(args.k * 100) 括号别错了

    print("---------------------------------------------------")
    if args.re == 1:
        raise NotImplementedError("ReOne Class")
        # graph = ReOne(dataPath, outputfolder, fracK, seed, use_fast_topndrop)
    else:
        graph = SSumM(dataPath, args.out, fracK, args.seed, args.use_fast_topndrop, args.urd, args.useNE)
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
