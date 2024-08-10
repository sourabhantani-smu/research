import numpy as np 
from numpy import linalg as la


verbose = 0
def chebyshev_filter(x, m, a, b, a0, A):
    e = (b-a)/2
    c = (b+a)/2
    s = e/(a0-c)
    s1=s
    y = (np.dot(A,x) - c*x)*(s1/e)
    for i in range(1,m):
        s_new = 1/((2/s1)-s)
        y_new = (np.dot(A,y) - c*y)*(2*s_new/e)-((s*s_new)*x)
        x=y
        y=y_new
        s=s_new
    return y

def swapIfNeeded(eigVal, eigVec):
    mu = eigVal[-1]
    i = eigVal.shape[0]-1
    maxI = i
    while mu < eigVal[i-1]:
        i -= 1
    sortedIndex=list(range(i))+[maxI]+list(range(i,maxI))
    eigVecSortedIndex = sortedIndex
    if eigVec.shape[0]>eigVal.shape[0]:
        eigVecSortedIndex = np.append(sortedIndex,np.array(range(len(eigVal), eigVec.shape[1])))
    return eigVal[sortedIndex], eigVec[:,eigVecSortedIndex], i!=maxI

def orthonormalizeDGKS(v, M):
    vnorm = la.norm(v)
    v = v/vnorm
    vnew_norm = None
    eta = 1/np.sqrt(2)
    rndVecIter = 0
    dgksiter = 0
    tmpnorm = eta*vnorm
    while vnew_norm is None or vnew_norm < tmpnorm:
        v_tmp = M.transpose().dot(v) 
        tmpnorm = la.norm(v_tmp)
        v_new = v - M.dot(v_tmp)
        vnew_norm = la.norm(v_new)
        v_new = v_new/vnew_norm
        dgksiter += 1
        if dgksiter > 3:
            print("max dgks iter reached:", vnew_norm, eta*vnorm)
            if rndVecIter < 3:
                rndVecIter += 1
                v =  np.matrix(np.random.randn(v.shape[0])).transpose()
            else:
                raise Exception("DGKS could not orthornormalize after 3 iterations and 3 random vectors")
        else:
            v = v_new
    return v_new


def lanczos_upperb(A, k=4):
    dim = A.shape[0]
    if k == -1:
        k=dim
    T = np.matrix(np.zeros((k, k), dtype=float))

    v_arr = []
    v = np.matrix(np.random.randn(A.shape[0])).transpose()
    v = v/la.norm(v)
    w_prime = A.dot(v)
    a = v.transpose().dot(w_prime)[0,0]
    w_new = w_prime - a*v
    v_arr.append(v)
    T[0,0]=a
    for i in range(1,k):
        b=la.norm(w_new)
        if b!=0:
            v_new = w_new/b
            v_arr.append(v_new)
        else:
            v_arr = generate_linearly_independent_unit_vector(v_arr)
            v_new = np.matrix(v_arr[-1]).transpose()
        w_prime = A.dot(v_new)
        a = v_new.transpose().dot(w_prime)[0,0]
        w_new = w_prime - a*v_new - b*v
        v=v_new
        T[i,i] = a
        T[i-1, i] = b
        T[i,i-1] = b
    max_eig = max(la.eig(T[:i+1, :i+1])[0])
    if b < 0.01:
        return max_eig+b*10
    elif b < 0.1:
        return max_eig+b*5
    else:
        return max_eig+b
    
    
def getUpperBound(A):
    nrm_1 = la.norm(A,1)
    return min(nrm_1, lanczos_upperb(A))
    
def cheb_dav(A, kwant, x = None, m=10, kkeep=None, maxdim=None, tol=1e-10, maxIter = 1e3):
    if x is None:
        x = np.matrix(np.random.randn(A.shape[0])).transpose()
    if kkeep is None:
        kkeep = kwant*2
    if maxdim is None:
        maxdim=kwant*4
        
    eigs = np.array([])
    x = x/la.norm(x)
    V = np.matrix(x)
    w = A.dot(x)
    W = np.matrix(w)
    mu = x.transpose().dot(w)[0,0]
    H = np.matrix(mu)
    
    r = w - mu*x
    if la.norm(r,2) <= tol:
        eigs = np.append(eigs, mu)
        kc = 1
        H = np.empty((0,0))
    else:
        kc=0
    upperb = getUpperBound(A)
    lowerb = (upperb + mu)/2
    a0 = lowerb
    ksub = 0
    itercount=0
    while itercount < maxIter:
        if verbose > 1:
            print('Upperbound', upperb)
            print('Lowerbound', lowerb)
        t = chebyshev_filter(x,m,lowerb, upperb, a0, A)
        v_knext = orthonormalizeDGKS(t,V[:,0:ksub+1])
        V = np.column_stack([V, v_knext])
        kold = ksub
        ksub += 1
        w_knext = A.dot(v_knext)
        W = np.column_stack([W,w_knext])
        if verbose > 1:
            print('V', V.shape,'W', W.shape,'H', H.shape)
            print('kc:',kc, ',    ksub:',ksub, ',     kold:',kold)        
        #H=np.row_stack([H[:ksub-kc,:ksub-kc],np.zeros(ksub-kc)])
        H=np.row_stack([H,np.zeros(kold+1)])
        if kc>0:
            H=np.column_stack([H,np.row_stack([np.zeros((kc,1)), V[:,kc:ksub+1].transpose().dot(w_knext)])])
        else:
            H=np.column_stack([H,V[:,kc:ksub+1].transpose().dot(w_knext)])
        for i in range(H.shape[0]-1):
            H[-1,i]=H[i,-1]
        D,Y = la.eigh(H[kc:ksub+1, kc:ksub+1])
        if verbose > 1:
            print('Y',Y)
            print('D',D)
        mu = D[0]
        if ksub >= maxdim:
            if verbose > 0:
                print("INNER RESTART")
            ksub = kc+kkeep
            #H = H[:ksub, :ksub]
            #V = V[:,:ksub]
        if verbose > 0:
            print('kc',kc, '      kold',kold,'         ksub',ksub)
        if kc > 0:
            V = np.column_stack([V[:,:kc], V[:,kc:kold+2].dot(Y[:,:ksub-kc+1])])
            W = np.column_stack([W[:,:kc], W[:,kc:kold+2].dot(Y[:,:ksub-kc+1])])
        else:
            V = np.column_stack([V[:,:kc], V[:,kc:kold+2].dot(Y[:,:ksub-kc+1])])
            W = np.column_stack([W[:,:kc], W[:,kc:kold+2].dot(Y[:,:ksub-kc+1])])

        if kc>0:
            H = np.column_stack([np.row_stack([H[:kc,:kc],np.zeros((ksub+1-kc,kc))]),np.row_stack([np.zeros((kc,ksub+1-kc)),np.diag(D[:ksub-kc+1])])])
        else:
            H = np.diag(D[:ksub+1])
        r = W[:,kc]-mu*V[:,kc]
        if verbose > 0:
            print('2-norm r',la.norm(r,2))

        itercount +=1
        for conv_i in range(len(D)):
            if verbose > 1:
                print(f'convergenceCheck: norm(r): {la.norm(r,2)}, {tol*max(D[conv_i:])}, V {V.shape}, W {W.shape}, H {H.shape}, norm(H) {la.norm(H,2)}') 
            if la.norm(r,2) <= tol*max(D[conv_i:]):
                kc += 1
                mu = D[conv_i]
                eigs = np.append(eigs,mu)
                if verbose > 0: 
                    print(f"CONVERGED: #{eigs.shape[0]}, in {itercount} steps:",mu)
                ## SWAP TEST and set swap=True if swap happens
                swap = False
                if kc > 1:
                    eigs, V, swap = swapIfNeeded(eigs, V)
                    
                if kc >= kwant and not swap:
                    return eigs, V[:,:kc+1]
                mu = D[conv_i+1]
                if kc < V.shape[1]:
                    r = W[:,kc]-mu*V[:,kc]
            else:
                break
        lowerb = np.median(D[:ksub])
        if a0 > min(D):
            a0 = min(D)
        x = V[:,kc]
