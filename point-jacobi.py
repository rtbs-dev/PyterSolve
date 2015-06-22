def PJ(phi0,f,n,xi,xf,its, dim='1d', graph=True, ax=None):
    
    if dim=='1d':
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
        phi[len(phi)-1]=0   #ensure bdry conditions at end
        
        b= [h**2*f(i) for i in x]
        
        for k in range(its):
            phinew=[0]      #initialize storage vector at 0     
            for i in range(1,n-1):
                phinew.append(0.5*(phi[i-1]+phi[i+1]) - b[i])
                #print phinew,'**' debug
            phi=phinew+[0]
            
            if graph:
                
                if sct is not None:
                    del ax.lines[0]
                sct=ax.plot(x, phi, 'b')
                ax.set_xlabel('x')
                ax.set_ylabel('phi(x)')
                ax.set_title('Point Jacobi approximation after '+str(k)+' iterations')
                plt.pause(.002)
                #time.sleep(0.05)
        
        return phi
    
    elif dim=='2d':
        
        if graph:
            sct=None
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.pyplot as plt
            from matplotlib import cm
            import time
            import numpy as np
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
            plt.ion()
            plt.show()
        
        (xyi, xyf) = xi, xf 
        #this will work for x/y bdrys equal, with same mesh spacing
        
        h=(xyf-xyi)/float(n-1)
        x=[xyi+h*i for i in range(n)]
        y=[xyi+h*i for i in range(n)]
        phi=[[phi0(i,j) for i in x]for j in y]
        #print len(phi)#debug
        b= [[h**2*f(i,j) for i in x] for j in y]   
        #print b[1]
        for k in range(its):
            phinew=[n*[0]for i in range(n)]   #initialize storage vector at 0's    
            #print phinew #debug
            for j in range(1,n-1):
                for i in range(1,n-1):
                    #notice size diff...want phinew[0] and phi[1]
                    phinew[j][i]=0.25*(phi[i-1][j]+phi[i+1][j]+phi[i][j-1]+phi[i][j+1] - b[j][i])
                    #print phinew[0][i] #debug
            phi=phinew[:]
                #inserting phinew into center of phi
            if graph:
                plt.gca()
                if sct is not None:
                    sct.remove()
                X,Y=np.meshgrid(x,y)
                
                sct=ax.plot_surface(X, Y, phi, rstride=4, cstride=4, alpha=0.25) #
                #cset = ax.contour(X, Y, phi, zdir='z', offset=-1, cmap=cm.coolwarm)
                #cset = ax.contour(X, Y, phi, zdir='x', offset=-1.5, cmap=cm.coolwarm)
                #cset = ax.contour(X, Y, phi, zdir='y', offset=1.7, cmap=cm.coolwarm)
                ax.set_ylabel("y-axis")
                ax.set_xlabel("x-axis")
                ax.set_zlabel("phi(x,y)")
                
                ax.set_title('Point Jacobi approximation after '+str(k)+' iterations')
                ax.legend(loc=0)
                plt.pause(.002)
                #time.sleep(0.05)
                
        return phi
    
    
    else:
        assert ((dim=='2d') or (dim=='1d')), "PJ PyterSolve only supports 1d or 2d"