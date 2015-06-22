"""
Below are tools needed for the MultiGrid methods at the end of the file.
"""
import math as m
import numpy as np
def GS_list_1d(phi0,b,its): #Gauss seidel using a list-initial, not function
    phi=phi0[:]
    n=len(phi0)
    for k in range(its):
        for i in range(1,n-1):
            phi[i]=0.5*(phi[i-1]+phi[i+1] - b[i])
    return phi
    
def der2_2o_central_list(f,h): #with 1st order boundaries
    #NOW WITH SAME SIZE AS F!!!
    n = len(f)
    df=[(f[0]-2*f[1]+f[2])/float(h**2)] #bdry is fwd
    for i in range(1,n-1):
        df.append((f[i+1]-2*f[i]+f[i-1])/float(h**2))
    df.append((f[n-1]-2*f[n-2]+f[n-3])/float(h**2)) #bdry is backwd
    return df
    
def r_laplace_1d(approx,h): #calculates residual of approx. for 1D LAPLACE
    
    return np.subtract(len(approx)*[0],der2_2o_central_list(approx,h))
    

def prolong_1d(nums):
    p_nums=nums[:] #copy existing, aligned meshpts
    
    for i in range(len(nums)-1,0,-1):
        #print p_nums[i], p_nums[i-1] #debug
        p_nums[i:i]=[0.5*(p_nums[i]+p_nums[i-1])] #insert interp from the top
    
    return p_nums

def restrict_1d(nums): #needs odd size of nums
    r_nums=[nums[0]]   
    for i in range(1,len(nums)/2):
        r_nums+=[0.25*(nums[2*i-1]+2*nums[2*i]+nums[2*i+1])]
    r_nums.append(nums[-1])
    return r_nums

"""
Below are the actual MG methods
"""

def DG(phi0, n, xi, xf, its, graph=True, ax=None):#initial state, fine mesh, start, end, iters
    #this dual-grid solver in 1D uses ONE G-S iteration on each level, with 
    #non-averaged restriction.
    if graph:
        sct=None
        import matplotlib.pyplot as plt
        import numpy as np
        if ax is None:
            fig = plt.figure()
            ax=fig.add_subplot(111)
        plt.ion()
        plt.show()
    h=(xf-xi)/float(n-1)
    x=[xi+h*i for i in range(n)]
    phi=[phi0(i) for i in x]    
    phi[len(phi)-1]=0.
    #phitop=[phi0(i) for i in x]
    for k in range(its):
        phi=GS_list_1d(phi,len(phi)*[0],1)[:]
        r_fine=r_laplace_1d(phi,h)[:]
        r_course=r_fine[::2] #every other point; interior
        
        err_course=GS_list_1d(len(r_course)*[0],r_course*(2*h)**2,1)[:]
        err_fine = prolong_1d(err_course[:])
        phi= np.add(phi,err_fine)
        if graph:
            if sct is not None:
                del ax.lines[0]
            sct=ax.plot(x, phi, 'b')
            ax.set_xlabel('x')
            ax.set_ylabel('phi(x)')
            ax.set_title('Dual-Grid approximation after '+str(k)+' iterations')
            plt.pause(.002)
            #time.sleep(0.05)
        
        
        
def MGv(phi0,n,xi,xf,its, graph=True, ax=None):
    #this is a V-cycle for n=2^k+1 mesh points. 
    if graph:
        sct=None
        import matplotlib.pyplot as plt
        import numpy as np
        if ax is None:
            fig = plt.figure()
            ax=fig.add_subplot(111)
        plt.ion()
        plt.show()
    p = int(m.log(n-1,2)) #the number of levels
    h=(xf-xi)/float(n-1)
    x=[xi+h*i for i in range(n)]
    phi=[phi0(i) for i in x]
    phi[len(phi)-1]=0.

    hlist=[h*2**i for i in range(p)] #list of mesh sizes
    for k in range(its):
        
        phi=GS_list_1d(phi,len(phi)*[0],1)[:] #top level
        res=r_laplace_1d(phi,hlist[0])[:] #top residual
        
        res_list=[] #keep a list of residuals for each level
        err_list=[] #list of epsilons for each level
        for q in range(1,p):
            res_new=restrict_1d(res)[:]
            err=GS_list_1d(len(res_new)*[0],[i*(hlist[q])**2 for i in res_new],1)[:]
            err_list.append(err)
            res=np.subtract(res_new,der2_2o_central_list(err,hlist[q]))[:]
            res_list.append(res)
            #print q
            
        for q in range(p-3,-1,-1):
            err_new = prolong_1d(err)[:]
            err = list(np.add(err_new,err_list[q])[:])
            err=GS_list_1d(err,res_list[q]*hlist[q]**2,1)[:]
            #print q
        err=prolong_1d(err)[:]
        phi= np.add(phi,err)
        if graph:
            if sct is not None:
                del ax.lines[0]
            sct=ax.plot(x, phi, 'b')
            ax.set_xlabel('x')
            ax.set_ylabel('phi(x)')
            ax.set_title('V-cycle approximation after '+str(k)+' iterations')
            plt.pause(.002)
            #time.sleep(0.05)
    return phi
        
        
        