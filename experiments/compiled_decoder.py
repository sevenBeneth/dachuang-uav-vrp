"""补充实验使用的等价 split 内核：只编译循环，不改变递推与平局规则。"""
import numpy as np
from numba import njit
from uavvrp.drone import DroneProblem


@njit(cache=True)
def split_kernel(table, kmax):
    n=table.shape[0]-1
    dp=np.full((n+1,kmax+1),np.inf)
    parent=np.full((n+1,kmax+1),-1,np.int64)
    dp[0,0]=0.
    for kk in range(1,kmax+1):
        for j in range(1,n+1):
            best=np.inf;best_i=-1
            for i in range(j-1,-1,-1):
                base=dp[i,kk-1]
                if base==np.inf:continue
                segment=table[i,j]
                if segment==np.inf:continue
                value=base+segment
                if value<best:best=value;best_i=i
            dp[j,kk]=best;parent[j,kk]=best_i
    return dp,parent


class CompiledDroneProblem(DroneProblem):
    def decode(self,perm,k_requested):
        n=len(perm);kmax=min(self.max_routes,n)
        dp,parent=split_kernel(np.asarray(self._segment_table(perm)),kmax)
        attainable=[k for k in range(1,kmax+1) if dp[n,k]<np.inf]
        if not attainable:return None
        used=min(attainable,key=lambda k:(abs(k-k_requested),k))
        routes=[];j,k=n,used
        while k>0:
            i=int(parent[j,k]);routes.append(list(perm[i:j]));j,k=i,k-1
        return routes[::-1],used


def warmup():
    split_kernel(np.array([[np.inf,1.],[np.inf,np.inf]]),1)
