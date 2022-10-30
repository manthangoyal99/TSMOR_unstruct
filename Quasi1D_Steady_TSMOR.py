# This program implements the Transported Snapshot Model Order Reduction method
# of Nair and Balajewicz (2019), specifically for the Quasi-1D Steady C-D nozzle
# flow problem (problem #1 of their work)
#need to change this
from cProfile import label
from cmath import nan
from traceback import print_tb
#from tkinter import Y
import numpy as np
import matplotlib.pylab as plt
from math import pi as mathPi
from math import pow as mathPow
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from scipy.interpolate import SmoothBivariateSpline
import copy
from MyPythonCodes.tools import Findiff_Taylor_uniform
from MyPythonCodes.mesh import UnstructuredMesh, getFileExtUSMD, \
    meshCellFaceProps,PolygonNormAreaCntrd
from scipy.spatial import KDTree
from pysph.base.utils import get_particle_array
from pysph.tools.interpolator import Interpolator
import gc
from SteadyAeroDyn_PY.SteadyAeroDyn.DataHandling import IntegralHlprDomainGrad
import timeit

'''
check if interpolation is happening properly
see if a surface equation can be obtained easily

'''
#np.set_printoptions(threshold=np.inf)
lamb = 0.01

def calc_grid_distortion_basis(points,dMu,ib,nf,nf_dep,axis):
    """
    Calculate one particular basis function towards the grid distortion.
    Referring to eqn. 11 of Nair & Balajtewicz (2019), this returns f_p*g_q for
    one p-q pair, towards the grid distortion c_s.
    
    INPUTS:
    points : 2d array of grid points suchh that indices represent the node and columns are the co-ordinates
    dMu  : Parameter difference between reference and predicted snapshots
    ib   : Basis function index ('m' in eqn. 11 above but with 0-based indexing)
    nf   : Total number of 'f' basis functions ('N_p' in eqn. 11 above)
    
    OUTPUTS:
    fpgq : Product of f_p & g_q evaluated over the grid 'xarr'
    """
    ibf = ib % nf       #'f' basis function term's index corresponding to 'ib'
    #print(ib,"\n")
    #print(ibf,"\n")
    ibg = ib//nf
    #print(ibg,"\n")        #'g' basis function term's index corresponding to 'ib'
    xi = 0        #Starting point of x-grid
    yi = 0        #Starting point of y-grid
    Lx = np.max(points[:,0])-xi  #Range of x-grid
    Ly = np.max(points[:,1])-yi  #Range of x-grid
    # Calculate the 'f_p' factor towards the basis, where 'p' is the index 'ibf'
    
    if(axis=='x'):
        if ibf == 0:
            grid_distortion_basis_term_f = np.ones(np.shape(points)[0])
        elif ibf<nf-nf_dep:
            grid_distortion_basis_term_f = np.sin(ibf*np.pi*(points[:,0]-xi)/Lx)
        #grid_distortion_basis_term_f = np.sin((ibf+1)*np.pi*(points[:,0]-xi)/Lx)
        else:
            grid_distortion_basis_term_f = np.sin((ibf-(nf-nf_dep)+1)*(np.pi)*(points[:,1]-yi)/Ly)

    else:
        if ibf == 0:
            grid_distortion_basis_term_f = np.ones(np.shape(points)[0])
        elif ibf<=nf_dep:
            grid_distortion_basis_term_f = np.sin(ibf*(np.pi)*(points[:,0]-xi)/Lx)
        else:
            grid_distortion_basis_term_f = np.sin((ibf-nf_dep)*np.pi*(points[:,1]-yi)/Ly)
        

    # Calculate the 'g_q' factor towards the basis, where 'q' is the index 'ibg'
    grid_distortion_basis_term_g = mathPow(dMu,ibg+1)
    # Return the product of 'f_p' and 'g_q'
    #print("shape",np.shape(grid_distortion_basis_term_f*grid_distortion_basis_term_g))
    return grid_distortion_basis_term_f*grid_distortion_basis_term_g
#enedef calc_grid_distortion_basis


def calc_grid_distortion(coeffsx,coeffsy,points,dMu,nfxy,nfyx,ng=1):
    """
    Calculate the grid distortion.
    Referring to eqn. 11 of Nair & Balajewicz (2019), this returns c_s.
    
    INPUTS:
    coeffs : Array of coefficients towards the grid distortion
    xarr   : Array of grid points
    dMu    : Parameter difference between reference and predicted snapshots
    ng     : Total number of 'g' basis functions ('N_q' in eqn. 11 above)
    
    OUTPUTS:
    cs : Sum-product of grid distortion basis functions and coefficients
    """
    nfx = len(coeffsx)/ng
    nfy = len(coeffsy)/ng
    grid_distortion = np.zeros_like(points)

    for ib, cc in enumerate(coeffsx):
        #print(cc,"cc")
        #print(np.shape(grid_distortion))
        grid_distortion[:,0] += cc*calc_grid_distortion_basis(points,dMu,ib,nfx,nfxy,'x')
    for ib, cc in enumerate(coeffsy):
        grid_distortion[:,1] += cc*calc_grid_distortion_basis(points,dMu,ib,nfy,nfyx,'y')
    return grid_distortion
#enddef calc_grid_distortion


def calc_distorted_grid(coeffsx,coeffsy,points,dMu,nfxy,nfyx,ng=1):
    """
    Calculate the distorted grid.
    Referring to eqn. 7 of Nair & Balajewicz (2019), this returns xd = x + c_s.
    
    INPUTS:
    coeffs : Array of coefficients towards the grid distortion
    xarr   : Array of grid points
    dMu    : Parameter difference between reference and predicted snapshots
    ng     : Total number of 'g' basis functions ('N_q' in eqn. 11 above)
    
    OUTPUTS:
    xd : Distorted grid (supplied original grid 'xarr' plus the grid distortion
         'c_s')
    """
    print("/home/manthangoyal/DDP_codes/TSMOR_UNSTRUCT/Quasi1D_Steady_TSMOR.py/Plotting in calc_distorted_grid")
    points_new = points + calc_grid_distortion(coeffsx,coeffsy,points,dMu,nfxy,nfyx,ng=ng)
    plt.scatter(points[:,0],points[:,1],c="r",label = "original",s=2)
    plt.scatter(points_new[:,0],points_new[:,1],c="b",label = "Dist",s=2)
    plt.legend()
    plt.show()
    return points + calc_grid_distortion(coeffsx,coeffsy,points,dMu,nfxy,nfyx,ng=ng)
#enddef calc_distorted_grid


def calc_transported_snap(coeffsx,coeffsy,points,u,dMu,nfxy,nfyx,ng=1):
    """
    Calculate the 'u' of LHS of eqn. 7 of Nair & Balajewicz (2019), by
    interpolating the given snapshot 'u' from the original grid to the distorted
    grid
    
    INPUTS:
    coeffs : Array of coefficients towards the grid distortion
    xarr   : Array of grid points
    u      : Reference snapshot; 2D array with rows corresponding to x-grid and
             columns corresponding to different components (flow variables)
    dMu    : Parameter difference between reference and predicted snapshots
    ng     : Total number of 'g' basis functions ('N_q' in eqn. 11 of Nair and
             Balajewicz (2019))
    
    OUTPUTS:
    ut : Transported version of given snapshot
    """
    # Distorted grid corresponding to given original grid, distortion
    # coefficients and parameter differential
    #print("coeffsx ", coeffsx)
    points_new = calc_distorted_grid(coeffsx,coeffsy,points,dMu,nfxy,nfyx,ng=ng)
    # We have to allow 'u' to be a 1D array corresponding to a single-component
    # (scalar) field; to make the subsequent steps agnostic to this, we reshape
    # it to a dummy 2D array if it is 1D in the following
    ushp = u.shape
    
    ##print(ushp)
    if len(ushp) > 1:   #2D array
        nc = ushp[1]
    else:               #1D array
        nc = 1
        u = np.reshape(u,(-1,1))    #Reshape 'u' to a single-column 2D array
    # Allocate return variable
    ut = np.zeros((ushp[0],nc)) #For now, this is a 2D array (reshaped later)
    start = timeit.default_timer()
 
    for ic in range(nc):    #Go thru each component
        # Generate the interpolation object by assuming that 'u' is specified
        # on the distorted grid
        ut[:,ic] = griddata(points_new,u[:,ic],points,method = 'linear',fill_value=nan)
        #extrap_func = KDTree(points_new)
        #print('nan values',np.size(np.argwhere(np.isnan(ut[:,ic]))))
        #x = np.argwhere(np.isnan(ut[:,ic])) 
        
        #print(np.shape(points_new))
        '''
        use when only 1 neighbour is required
        ''' 
        
        #dis,index = extrap_func.query(points[np.isnan(ut[:,ic])],1)
        #print(np.shape(u[index,ic]))
        #ut[np.isnan(ut[:,ic]),ic] = u[index,ic]
        
        '''
        use when more than 1 nbr is required 
        '''
        #dis,index = extrap_func.query(points[np.isnan(ut[:,ic])],2)
        #dis_copy = np.copy(dis)
        #dis_copy[dis[:,0]==0,0] = 1e-20
        #print(np.shape(dis))
        ##print(dis)
        #weights = (1/dis_copy)/(np.sum(1/dis_copy,axis = 1,keepdims=True))
        ##print("weights",(weights))
        #print('0 values',np.size(np.argwhere(dis[:,0]==0)))
        #print(np.shape(u[index,ic]))
        #extrap_data = np.sum(weights*u[index,ic],axis = 1)
        ##print(np.shape(extrap_data))
        #ut[np.isnan(ut[:,ic]),ic] = extrap_data
        ##ut[index[dis[:,0]==0],ic] = u[index[dis[:,0]==0],ic]
        
        '''
        Pysph memthod for extrapolation
        '''

        h = 4 * np.max(np.diff(points_new[:,0],axis=0))
        m = h**2
        pa = get_particle_array(name = "myprop",x=points_new[:,0],y=points_new[:,1],density=u[:,0],h = 0.4*(points[0,0]-points[1,0]),
            m=m)
        interp = Interpolator([pa], x=points[np.isnan(ut[:,ic]),0],
                                  y=points[np.isnan(ut[:,ic]),1],method='shepard')
        
        ut[np.isnan(ut[:,ic]),ic] = interp.interpolate('density')


        #extrap_func = SmoothBivariateSpline(points_new[:,0],points_new[:,1],u[:,ic],s=0.0,kx=1,ky=1)
        ##print("indices ",np.argwhere(np.isnan(ut)))
        #print(np.shape(points_new[:,0]))
        #print(ut[np.isnan(ut[:,ic]),ic])
        #print(np.shape(ut[np.isnan(ut[:,ic]),ic]))
        #print(extrap_func(points[np.isnan(ut[:,ic]),0],points[np.isnan(ut[:,ic]),1]))
        #print(np.shape(extrap_func(points[:,0],points[:,1])))
        #ut[np.isnan(ut[:,ic]),ic] = extrap_func(points[np.isnan(ut[:,ic]),0],points[np.isnan(ut[:,ic]),1])
    stop = timeit.default_timer()


    print('Time: ', stop - start) 
    return np.reshape(ut,ushp)  #Make sure to return 1D array if input was so
#enddef calc_transported_snap


def calc_transported_snap_error(coeffsx,coeffsy,mesh_base,uRef,uNbs,dMuNbs,nfxy,nfyx,ref_error=None,ng=1):
    """
    Calculate the square of the 2-norm of error between the transported versions
    of a reference snapshot and other (neighbouring) snapshots
    
    INPUTS:
    coeffs : Array of coefficients towards the grid distortion
    xarr   : Array of grid points
    uRef   : Reference snapshot; 2D array with rows corresponding to x-grid and
             columns corresponding to different components (flow variables)
    uNbs   : List of neighbouring snapshots that should be predicted; each list
             list entry is a 2D array of the same shape as 'uRef'
    dMuNbs : Set of parameter differences between the reference snapshot and the
             neighbouring snapshots
    ng     : Total number of 'g' basis functions ('N_q' in eqn. 11 of Nair and
             Balajewicz (2019))
    
    OUTPUTS:
    error : Total error across all neighbouring snapshots
    """
    error = 0 #Initialize as 0 to calculate as running sum over all neighbours
    mesh_distorted = copy.deepcopy(mesh_base)

    for inb, dMu in enumerate(dMuNbs): #Go thru all neighbours
        # Transport the reference snapshot 'uRef' by the parameter differential
        # 'dMu' using the grid distortion coefficients 'coeff' over the original
        # grid 'xarr'
        u0t = calc_transported_snap(coeffsx,coeffsy,mesh_base.getNodes(),uRef,dMu,nfxy,nfyx,ng=ng)
        # Add the square of the 2-norm of the difference between the transported
        # snapshot and the neighbour to the running sum of error 
        #print("marker tags", mesh_base.getMarkTags())
        #nodes_upper = mesh_base.getMarkTags('upperwall')
        # points = mesh_base.getNodes()
        # plt.tricontour(points[:,0],points[:,1],uNbs[inb][:,0],20,colors = "black",linestyles = 'dashdot') 
        # plt.tricontour(points[:,0],points[:,1],u0t[:,0],20) 
        # #plt.savefig("/home/manthangoyal/manthan/study/TSMOR_data/x-4,2,y-2,0/unstruct/density/bump-points-constraint/shepard/"+str(self.iRef)+"/contour.png")
        #plt.show()
        mesh_distorted.nodes = calc_distorted_grid(coeffsx,coeffsy,mesh_base.getNodes(),dMu,nfxy,nfyx,ng=ng)
        nodes_lower = np.unique(mesh_base.getMarkCells('lowerwall')[0][0][0])
        nodes_lower_dist = np.unique(mesh_distorted.getMarkCells('lowerwall')[0][0][0])
        coor_lower = mesh_base.getNodes()[nodes_lower]
        dist_lower = (coor_lower[:][0]**2+coor_lower[:][1]**2)**.5
        coor_lower_dist = mesh_distorted.getNodes()[nodes_lower_dist]
        dist_lower_dist = (coor_lower_dist[:][0]**2+coor_lower_dist[:][1]**2)**.5
        #print((nodes_lower))
        #print((type(nodes_lower)))
        #exit()
        #error += np.linalg.norm(u0t - uNbs[inb])**2 +lamb*(np.linalg.norm(u0t[nodes_lower]-uNbs[inb][nodes_lower])**2)
        error += np.linalg.norm((u0t - uNbs[inb]).flatten(),2)**2
        #error += np.linalg.norm(u0t - uNbs[inb])**2/2.484 +lamb*(np.linalg.norm(dist_lower-dist_lower_dist)**2)
        #exit()
    if ref_error==None: 
        return error
    else:
        return (error)/ref_error
#enddef calc_transported_snap_error


def calc_grid_distortion_constraint(coeffsx,coeffsy,mesh_base,dMuNbs,nfxy,nfyx,ng=1):
    """
    Inequality constraint to be satisfied by the transport field coefficients
    
    Essentially, we do not want the distorted grid to collapse (i.e., have zero
    or negative spacing) anywhere for any neighbour. To this end, we impse 2 conditions:
    1- Cell area should not become 0
    2- Cell normal should remain same

    The inequality constraint is
    
    cel_area>0 for all cells in the distorted mesh
    cell_base_normal * cell_distorted_normal > 0
    The optimizer takes inequality constraints of the form
        cieq(coeffs) < 0,
    where 'cieq' takes the coefficients' array (array of quantities to be
    optimized) and returns a list of values.
    
    INPUTS:
    coeffs : Array of coefficients towards the grid distortion
    xarr   : Array of grid points
    dMuNbs : Parameter differences between neighbouring and reference snapshots
    ng     : Total number of 'g' basis functions ('N_q' in eqn. 11 of Nair and
             Balajewicz (2019))
    
    OUTPUTS:
    ineq : List of inequality constraint values at all grid points and for all
           neighbours
    """
    # We arbitrarily specify the minimum allowed spacing of the distorted grid
    # as a small sub-multiple of the minimum grid spacing in the original
    # (undistorted) grid

    mesh_distorted = copy.deepcopy(mesh_base)
    # Allocate array of inequality constraint values to be returned
    mesh_base._readMeshCellNodes()
    ineq = np.zeros((mesh_base.getNCell(),len(dMuNbs)))
    nodes_lower = np.unique(mesh_base.getMarkCells('lowerwall')[0][0][0])
    coor_lower = mesh_base.getNodes()[nodes_lower]
    dx = np.sum(np.diff(coor_lower[:,0]))/np.size(coor_lower)
    #print('dx',dx)
    ineq_bump_sliding = np.zeros((np.shape(coor_lower)[0],len(dMuNbs)))
    #ineq_bump_sliding = []
    ### make a new ineq array which has data related to cell area and normal no. of cells * 2
    parameters_base = meshCellFaceProps.CalcSignedArea2d(mesh_base)
    area_min = np.min(parameters_base**2)/10000
    for iMu, dMu in enumerate(dMuNbs):

        #calculate the constraints to restrics excessive distortion
        mesh_distorted.nodes = calc_distorted_grid(coeffsx,coeffsy,mesh_base.getNodes(),dMu,nfxy,nfyx,ng)
        parameters_distorted = meshCellFaceProps.CalcSignedArea2d(mesh_distorted)
        ineq[:,iMu] = area_min -1*parameters_distorted*parameters_base

        #calculate the constraints to implement sliding boundary
        #nodes_lower_dist ->Nodes of the bump wall of the distorted mesh 
        nodes_lower_dist = np.unique(mesh_distorted.getMarkCells('lowerwall')[0][0][0]) 
        coor_lower_dist = mesh_distorted.getNodes()[nodes_lower_dist]
        #print(type(nodes_lower_dist))

        #nodes_lower_dist = nodes_lower_dist[np.where((coor_lower_dist[:,0]<=2) & (coor_lower_dist[:,0]>=1))]
        #coor_lower_dist = mesh_distorted.getNodes()[nodes_lower_dist]

        #print("number of points on boundary- ",np.shape(coor_lower,"  ", np.shape(coor_lower_dist)))
        extrap_func = KDTree(coor_lower)
        dis,index = extrap_func.query(coor_lower_dist,2)    
        dis_copy = np.copy(dis)
        dis_copy[dis[:,0]==0,0] = 1e-20
        weights = (1/dis_copy)/(np.sum(1/dis_copy,axis = 1,keepdims=True))
        x_avg = np.sum(weights*coor_lower[index,0],axis = 1,keepdims=True)
        y_avg = np.sum(weights*coor_lower[index,1],axis = 1,keepdims=True)
        r_avg = np.hstack((x_avg,y_avg))
        diff_vector = coor_lower_dist - r_avg
        vect1 = coor_lower_dist-coor_lower[index[:,0],:]
        vect2 = coor_lower[index[:,1],:]-coor_lower[index[:,0],:]
        vect3 = np.cross(vect1,vect2)
        dis_index_pts = (vect2[:,0]**2+vect2[:,1]**2)**.5
        ineq_bump_sliding[:,iMu] = np.abs(vect3/dis_index_pts)**2-0.3*dx
        #ineq_bump_sliding[:,iMu] = diff_vector[:,0]**2+diff_vector[:,1]**2 - .3*dx

    #return np.append(np.reshape(ineq,(-1),'F'),np.append(ineq_bump_sliding[0],ineq_bump_sliding[1]))
    return np.append(np.reshape(ineq,(-1),'F'),np.reshape(ineq_bump_sliding,(-1),'F'))
    #return np.reshape(ineq,(-1),'F')
#enddef calc_grid_distortion_constraint


class Project_TSMOR_Offline(object):
    """
    Starts an optimization project for offline part of TSMOR of the 1-D problem
    of Nair & Balajewicz (2019)
        
    ATTRIBUTES:
    uRef   : Reference snapshot that should be transported for predicting the
             following neighbouring snapshots; 2D array with rows corresponding
             to x-grid and columns corresponding to different components (flow
             variables)

    uNbs   : List of neighbouring snapshots that should be 'predicted'; each
             entry is a 2D array of the same shape as 'uRef'

    dMuNbs : Parameter differences between the reference snapshot and the above
             neighbouring snapshots
    mesh_base   : Array of grid points
    ng     : Total number of 'g' basis functions ('N_q' in eqn. 11 of Nair and
             Balajewicz (2019))
         
    METHODS:
    Optimizer interface - The following methods take a design variable vector 
    (coefficients of grid distortion bases) for input as a list (shape n) or
    numpy array (shape n or nx1 or 1xn); values are returned as float or list
    or list of list
    obj_f     - objective function              : float
    obj_df    - objective function derivatives  : list
    con_ceq   - equality constraints            : list
    con_dceq  - equality constraint derivatives : list[list]
    con_cieq  - inequality constraints          : list
    con_dcieq - inequality constraint gradients : list[list]
    Auxiliary methods -
    solnPost - Post-process optimization solution
    """  
    
    def __init__(self,u_dbs,Mus_db,iRef,iNbs,mesh_base,nfx,nfy,nfxy,nfyx,ng=1):
        """
        Constructor of class
        
        INPUTS:
        u_db    : 3D array of snapshot database, with dimensions as:
                   1: grid points
                   2: components (flow variables)
                   3: snapshots
        Mus_db : Parameters of the snapshots in the database
        iRef   : Index of reference snapshot to be transported
        iNbs   : Indices of neighbouring snapshots to be predicted
        xarr   : The base unstructured mesh with no distortion
        ng     : Total number of 'g' basis functions ('N_q' in eqn. 11 of Nair
                 and Balajewicz (2019))
        normalised : bool variable telling is normalised variables are used or not 
        """
        
        self.u_db_all = u_dbs
        self.u_ref_all = u_dbs[iRef,:,:]
        self.uNbs_all = [u_dbs[iNb,:,:] for iNb in iNbs]

        u_db = u_dbs[:,:,6:7] #using pressure as the only variable
        self.iRef = iRef
        #Extract reference snapshot
        self.uRef = u_db[iRef,:,:]    
        # Form list of neighbouring snapshots
        self.uNbs = [u_db[iNb,:,:] for iNb in iNbs]
        # Form list of corresponding parameter differentials from reference
        self.dMuNbs = [Mus_db[iNb]-Mus_db[iRef] for iNb in iNbs]
        self.ng = ng  #Store no. of 'g' basis function
        self.mesh_base = mesh_base 
        self.mesh_base._readLite()
        self.nfx = nfx #Total number of x-basis functions
        self.nfy = nfy #Total number of y-basis functions
        self.nfxy = nfxy #No. of basis f-functions in x grid distortion dependent on y
        self.nfyx = nfyx #No. of basis f-functions in y grid distortion dependent on x
        #Ref error is calculated which is the difference ebeteen the ref snapshot and the neighbouring snapshot
        self.ref_error = calc_transported_snap_error(np.zeros(nfx*ng),np.zeros(nfy*ng),self.mesh_base,\
            self.uRef,self.uNbs,self.dMuNbs,self.nfxy,self.nfyx,ng=self.ng)

    # Evaluates the objective function based on the supplied set of independent
    # variables (coefficients towards grid distortion that are to be optimized)
    def obj_f(self,coeffs):
        # Evaluate the error in predicting the neighbouring snapshots by
        # transporting the reference snapshot, and return as a singleton list
        coeffsx = coeffs[:self.nfx*self.ng]
        coeffsy = coeffs[self.nfx*self.ng:]

        return [calc_transported_snap_error(coeffsx,coeffsy,self.mesh_base,self.uRef, \
            self.uNbs,self.dMuNbs,self.nfxy,self.nfyx,ref_error= self.ref_error,ng=self.ng)]
    # Evaluates inequality constraint vector based on supplied set of
    # independent variables (coefficients towards grid distortion that are to be
    # optimized)
    def con_cieq(self,coeffs):
        # Evaluate the set of inequality constraints to be satisfied by the grid
        # distortions required to predict each neighbour
        coeffsx = coeffs[:self.nfx*self.ng]
        coeffsy = coeffs[self.nfx*self.ng:]
        return calc_grid_distortion_constraint(coeffsx,coeffsy,self.mesh_base,self.dMuNbs, \
            self.nfxy,self.nfyx,ng=self.ng)

    """ All the remaining optimizer interfaces are left blank """
    def con_ceq(self,coeffs):
        return []
    
    def con_dceq(self,coeffs):
        return []
    
    def obj_df(self,coeffs):
        return []

    def con_dcieq(self,coeffs):
        return []
    
    # Post-process the optimization solution
    def solnPost(self,output):
        
        
        coeffs = output[0]
        coeffsx = coeffs[:self.nfx*self.ng]
        coeffsy = coeffs[self.nfx*self.ng:]
        print('\t\tcoeffs = ['+', '.join(['%0.4f'%c for c in coeffs])+']')
        cieqs = np.array(self.con_cieq(coeffs))
        print('\t\tmax(con) = %0.4f, min(con) = %0.4f'%(max(cieqs),min(cieqs)))

        points = self.mesh_base.getNodes()
        
        plt.figure()
        for iNb, dMu in enumerate(self.dMuNbs):
            dist_nb = calc_transported_snap(coeffsx,coeffsy,points,self.u_ref_all,dMu,self.nfxy,self.nfyx,ng=self.ng)
            #dist_nb_sq = np.reshape(dist_nb,(128,256,-1))
            #uNbs_sq = np.reshape(self.uNbs[iNb],(128,256,-1))
            plt.subplot(2,1,iNb+1)
            plt.tricontour(points[:,0],points[:,1],self.uNbs_all[iNb][:,3],20,colors = "black",linestyles = 'dashdot') 
            CS = plt.tricontour(points[:,0],points[:,1],dist_nb[:,3],20) 
            plt.clabel(CS, inline=1, fontsize=10)
        plt.title("Distorted and neighboouring snapshot")
        #plt.savefig("/home/manthangoyal/manthan/study/TSMOR_data/x-4,2,y-2,0/unstruct/density/bump-points-constraint/shepard/"+str(self.iRef)+"/contour.png")
        plt.show()

        for iNb, dMu in enumerate(self.dMuNbs):
            for i in range(np.shape(self.uRef)[1]):
                dist_nb = calc_transported_snap(coeffsx,coeffsy,points,self.uRef,dMu,self.nfxy,self.nfyx,ng=self.ng)
                cs1 = plt.tricontour(points[:,0],points[:,1],dist_nb[:,i],20)
                plt.colorbar()
                cs2 = plt.tricontour(points[:,0],points[:,1],self.uNbs[iNb][:,i],20,colors = "black",linestyles = 'dashdot')
                h1,_ = cs1.legend_elements()
                h2,_ = cs2.legend_elements()
                plt.legend([h1[0], h2[0]], ['Computed', 'Actual'])
                plt.xlabel('x')
                plt.ylabel('y')
                ax = plt.gca() 
                ax.set_aspect(1) 
                plt.title("Distorted and neighboouring snapshot")
                plt.show()
        
        #plt.savefig("/home/manthangoyal/manthan/study/TSMOR_data/x-4,2,y-2,0/unstruct/density/bump-points-constraint/shepard/"+str(self.iRef)+"/contour.png")
            
        plt.figure()
        for iNb, dMu in enumerate(self.dMuNbs):
            dist_nb = calc_transported_snap(coeffsx,coeffsy,points,self.uRef,dMu,self.nfxy,self.nfyx,ng=self.ng)
            plt.subplot(2,1,iNb+1)
            plt.scatter(points[:,0],points[:,1],c = abs(dist_nb[:,0]-self.uNbs[iNb][:,0]),s=7)
        plt.title("Error scatter plot")
        #plt.savefig("/home/manthangoyal/manthan/study/TSMOR_data/x-4,2,y-2,0/unstruct/density/bump-points-constraint/shepard/"+str(self.iRef)+"/error.png")
        plt.show()

        plt.figure()
        for iNb, dMu in enumerate(self.dMuNbs):
            #plt.subplot(2,1,iNb+1)
            #plt.plot(calc_distorted_grid(coeffs,self.xarr,dMu,self.normalised),self.uRef[:,iv]*self.u_db_inlet[iv,iRef],'--')
            points_new = calc_distorted_grid(coeffsx,coeffsy,points,dMu,self.nfxy,self.nfyx,ng=self.ng)
            plt.scatter(points[:,0],points[:,1],color = "blue",s=2,label='Points on the base mesh')
            plt.scatter(points_new[:,0],points_new[:,1],color = "red",s =2,label = 'Points on the distorted mesh')
            #plt.legend()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Distortions')
            plt.show()
        
        #plt.savefig("/home/manthangoyal/manthan/study/TSMOR_data/x-4,2,y-2,0/unstruct/density/bump-points-constraint/shepard/"+str(self.iRef)+"/distortions.png")
        
        #u0t = calc_transported_snap(coeffsx,coeffsy,mesh_base.getNodes(),uRef,dMu,nfxy,nfyx,ng=ng)
        nodes_lower = np.unique(self.mesh_base.getMarkCells('lowerwall')[0][0][0])

        for iNb, dMu in enumerate(self.dMuNbs):
            dist_nb = calc_transported_snap(coeffsx,coeffsy,points,self.uRef,dMu,self.nfxy,self.nfyx,ng=self.ng)
            plt.scatter(points[nodes_lower,0],dist_nb[nodes_lower,0],label = 'distorted')
            plt.scatter(points[nodes_lower,0],self.uNbs[iNb][nodes_lower,0],label='actual')
            plt.legend()
            plt.show()

        for iNb, dMu in enumerate(self.dMuNbs):
            dist_nb = calc_transported_snap(coeffsx,coeffsy,points,self.uRef,dMu,self.nfxy,self.nfyx,ng=self.ng)
            print(f"lower wall error{iNb}", np.linalg.norm(abs(dist_nb[nodes_lower,0]-self.uNbs[iNb][nodes_lower,0]))/
                        np.linalg.norm(self.uNbs[iNb][nodes_lower,0]))



                

#endclass Project_TSMOR_Offline


def calc_transported_basis(coeffsRefsx,coeffsRefsy,points,uRefs,dMuRefs,nfxy,nfyx,ng=1):
    """
    Calculate the basis set for a new case by transporting various reference
    snapshots using grid distortion coefficients precalculated in offline phase,
    and parameter differentials between the reference snapshots and the new case
    to be predicted
    
    INPUTS:
    coeffsRefsx : 2D array of coefficients towards grid distortions in x, with
                 dimensions as:
                 1: Various grid distortion basis functions
                 2: Various snapshots to be transported
    coeffsRefsy : 2D array of coefficients towards grid distortions in y, with
                 dimensions as:
                 1: Various grid distortion basis functions
                 2: Various snapshots to be transported
    points       : 2D Array of grid points [x,y]
    uRefs      : 3D array of reference snapshots to be 'transported', with
                 dimensions as:
                 1: data along nodes
                 2: data for each component (flow variable),
                 3: data for each reference snapshot
    dMuRefs    : Parameter differences between the reference snapshots and the
                 case to be predicted
    ng         : Total number of 'g' basis functions ('N_q' in eqn. 11 of Nair
                 and Balajewicz (2019))
    nfxy       : y dependency of x basis functions
    nfxy       : x dependency of y basis functions
    
    OUTPUTS:
    phis : List of transported reference snapshots that form the basis set for
           the new case
    """
    phis = []
    for iRef in range(len(dMuRefs)):
        phis.append(calc_transported_snap(coeffsRefsx[:,iRef],coeffsRefsy[:,iRef],\
            points,uRefs[iRef,:,:],dMuRefs[iRef],nfxy,nfyx,ng=ng))
    return phis
#enddef calc_transported_basis

def residual(mesh,u,uFOM,LNorm,sigma,refNorm=None):

        gamma_value = 1.4
        # Pre-allocate storage for normalized weighted residual vector (output);
        # it will be flattened prior to actual return
        ResN = 4
        NCellRes = mesh.getNCell()
        Res = np.zeros((ResN,NCellRes))
        dataType = 'N'
        NDime =2
        intHlprGrad = IntegralHlprDomainGrad(mesh,dataType)
        # We find the face flux values from (a) the flow variable values on the
        # faces, and (b) the pre-calculated face areas & their unit normals.
        # For node-based data, the average of the vertex values is deemed the
        # face value of the data.
        # Subsequently, the integrated cell divergences are calculated as the
        # dot product of the face fluxes with the face normal vectors, and the
        # results are added over all the faces.
        # To this, we add the pressure gradient for momentum residues.
        # N.B.: The algorithm doesn't need to make any special provision for the
        # various element types.
        ncrY = 0 #No. of Cells whose Residuals are found Yet (0 at start)
        for iTyp, FaceMeasuresTyp in enumerate(intHlprGrad.FaceMeasures):
            if FaceMeasuresTyp is None: #No cells of this type in mesh
                continue
            # Find No. of Cells whose Residuals are to be found for this Type
            ncrT = FaceMeasuresTyp.shape[0]
            nFacesInCellsOfTyp = FaceMeasuresTyp.shape[1]
            for ifc in range(nFacesInCellsOfTyp):#Go thru all faces of cell type
                # Extract the direction cosines and areas of this face in all
                # retained cells of this type
                faceDirCoss = FaceMeasuresTyp[:,ifc,:-1]
                faceAreas = FaceMeasuresTyp[:,ifc,-1]

                # In the following, we need density, pressure and temperature.
                # N.B. whereas density and temperature are normalized by their
                # respective ambient values (rho_inf & temp_inf), pressure is
                # normalized by rho_inf * a_inf**2 (for compatibility in the
                # momentum equation). Thus, the normalized ideal gas law is
                #   pressure = density * temperature / SpHt.
                
                # At this stage pressure, density and temperature are available;
                # these are all 1D arrays of same length
                dataLen = np.shape(u)[0]    #Get common length of data arrays
                faceContribs = intHlprGrad.faceContribs[iTyp][ifc]

                #Full-domain solution reconstructed in RAM
                # Get 2D array of node/face indices of mesh that contribute, with
                # rows being all cells of this type being considered, and columns
                # being the vertices of the face (for node-based data) or the
                # (singleton) face itself (for face-based data)
                iContribs = intHlprGrad.cellContribs[iTyp][:,faceContribs]
                #print(iContribs)
               
                #print(np.shape(u))
                # Retrieve momentum componets
                density = np.average(u[iContribs,0:1],axis=1)
                pressure = np.average(u[iContribs,3:4],axis=1)
                vels = np.average(u[iContribs,1:3],axis=1)
                #print(np.shape(vels))
                #print(np.shape(faceAreas))
                #print(np.shape(pressure[:]))
                #vels = np.stack((vel_x,vel_y),axis=1)
                momenta = density*vels
                temperature = pressure / density * gamma_value
                
                #print("hopefully values are correctly calculated")
                # Calculate mass flux on faces using momenta and face properties
                mf = np.sum(momenta*faceDirCoss,axis=1)*faceAreas
                # Compose the total enthalpy per unit mass
                #   enthalpyTot = Cp * T + (u**2+v**2+...)/2,
                # which is the sum of enthalpy + kinetic energies per unit mass.
                # Normalizing by a_inf**2, we have
                #print(temperature[:200,:])
                enthalpyTot = temperature.flatten()/(gamma_value-1)+np.sum(vels**2,axis=1)/2
                #print("done")
                #exit()
                # Calculate contribution of the current face to the residues of
                # the various equations in each cell (integrated over the 
                # corresponding cell volumes), and add to running sum over all
                # faces
                # Residue of mass cons. eqn. (w/ U = velocity vector [u, v, w])
                #   div(density U)
                Res[0,ncrY:ncrY+ncrT] += mf
                # Residue of energy conservation equation
                #    div(Etot density U)
                Res[3,ncrY:ncrY+ncrT] += enthalpyTot*mf
                # Residue of x-momentum conservation equation
                #    div(density u U) + dp_dx.
                # Likewise for the other two components.
                #print("done1")
                for iMom in range(NDime):
                    momContrib = pressure.flatten()*faceDirCoss[:,iMom]*faceAreas \
                        + mf*vels[:,iMom]
                    Res[1+iMom,ncrY:ncrY+ncrT] += momContrib
            #endfor ifc in range(nFacesInCellsOfTyp)
            # In case of L2-norm calculation, divide the residual in each cell
            # by the square root of its metric (volume in 3D; area in 2D; etc)
            # to get weighted residual, whose 2-norm will approximate L2-norm.
            # No such weighting is needed for 1-norm to approximate L1-norm.
            if LNorm == 2:
                for iRes in range(ResN):
                    Res[iRes,ncrY:ncrY+ncrT] \
                        /= np.sqrt(intHlprGrad.Metrics[iTyp])
            # Increment No. of Cells whose Residuals are found Yet with that of
            # Cells whose Residuals are found for this Type
            ncrY += ncrT
        #endfor iTyp, FaceMeasuresTyp in ...
        
        # Calculate the Lp^p norms of the individual (normalized) residuals
        ResNormsP = np.zeros((ResN))
        for iRes in range(ResN):
            ResNormsP[iRes] = np.linalg.norm(Res[iRes,:],ord=LNorm) \
                **LNorm        
        penalty = sigma*penalty_inlet(u,mesh,uFOM)
        #print(penalty)
        ResNormsP_penalty = ResNormsP+penalty
        #print(ResNormsP_penalty)

        #if refNorm is None:
        #    return np.linalg.norm(ResNormsP_penalty,ord=1)
        #else:
        #    #print(refNorm)
        #    #print(ResNormsP_penalty)
        #    return ResNormsP ,np.linalg.norm(ResNormsP_penalty,ord=1)/refNorm
        if refNorm is None:
            return ResNormsP_penalty
        else:
            #print(refNorm)
            #print(ResNormsP_penalty)
            return ResNormsP ,np.linalg.norm(ResNormsP_penalty/refNorm,ord=1)
    #enddef evalObj
        
def penalty_inlet(u,mesh,uFOM):

    mesh._readMeshCellNodes()
    nodes_inlet = np.unique(mesh.getMarkCells('inlet')[0][0][0])
    u_inlet = u[nodes_inlet,:]
    uFOM_inlet = uFOM[nodes_inlet,:]
    #print(u_inlet)
    #exit()

    penalNorm = np.zeros(np.shape(u)[1])
    for iRes in range(np.shape(u)[1]):
        penalNorm[iRes] = np.linalg.norm((u_inlet-uFOM_inlet)[:,iRes]\
            ,ord=2)**2        
    return penalNorm
    #u_error = np.linalg.norm(u_inlet-uFOM_inlet,2)**2

    #return u_error


class Project_TSMOR_Online(object):
    """
    Starts an optimization project for online part of TSMOR of the 1-D problem
    of Nair & Balajewicz (2019)
        
    ATTRIBUTES:
    prblm_setup : Parameters specifying quasi 1D flow problem thru C-D nozzle
    xarr        : Array of x-grid points
    sigma       : Scale factor for enforcing boundary conditions
    phis        : Basis set obtained by transporting given reference snapshots
                  for predicting new case
    coords0     : Initial guess of generalized coordinates for above basis set
    Dx          : 1st-order finite difference operator on the 'xarr' grid
         
    METHODS:
    Optimizer interface - The following methods take a design variable vector 
    (generalized coordinates for basis set of transported reference snapshots)
    for input as a list (shape n) or numpy array (shape n or nx1 or 1xn); values
    are returned as float or list or list of lists
    obj_f     - objective function              : float
    obj_df    - objective function derivatives  : list
    con_ceq   - equality constraints            : list
    con_dceq  - equality constraint derivatives : list[list]
    con_cieq  - inequality constraints          : list
    con_dcieq - inequality constraint gradients : list[list]
    Auxiliary methods -
    compose_soln - Compose the solution from generalized coordinates
    solnPost     - Post-process optimization solution
    """  
    
    def __init__(self,uTest,uRefs,MuRefs,coeffsRefsx,coeffsRefsy,mesh,MuNew, \
            coords0,nfx_y,nfy_x,ng=1,sigma=100000):
        # Register some of the supplied variables directly as attributes of self
        self.uTest = uTest ## It has density velx vely pressure [2d]
        self.mesh = mesh
        self.sigma = sigma
        self.all_res = []
        self.nodes = mesh.getNodes()
        self.nfxy = nfx_y
        self.nfyx = nfy_x
        self.ng = ng
        # Distances of new snapshot from reference ones in parameter space
        dMuRefs = MuNew - MuRefs
        # Basis calculation by transporting the supplied reference snapshots
        self.phis = calc_transported_basis(coeffsRefsx,coeffsRefsy,self.nodes,\
            uRefs,dMuRefs, self.nfxy,self.nfyx,ng=self.ng)
        # Initial guess of generalized coordinates (for basis set of transported
        # snapshots), based on inverse-distance weights in parameter space
        self.coords0 = coords0
        # Calculate 1st-order finite difference operator with 4th-order accuracy
        #self.Di = Findiff_Taylor_uniform(len(self.iarr),1,3)/(self.iarr[1]-self.iarr[0])
        #self.Dj = Findiff_Taylor_uniform(len(self.jarr),1,3)/(self.jarr[1]-self.jarr[0])
        u = self.compose_soln(self.coords0)
        self.refNorm = residual(self.mesh,u,self.uTest,2,self.sigma)
    # Compose the snapshot solution using the supplied set of generalized
    # coordinates that comprise the weights of the basis set 'phis'
    def compose_soln(self,coords):
        # Initialize 2D snapshot array as 0's, to be subsequently composed as a
        # running sum
        u = np.zeros_like(self.phis[0])
        for iphi in range(len(coords)): #Go thru each generalized coordinate
            u += self.phis[iphi]*coords[iphi] #Add contribution of this basis
        return u
    
    # Evaluates the objective function (l_1 norm of residual of governing 
    # equations, augmented by scaled discrepancies in boundary conditions) based
    # on the supplied generalized coordinates
    def obj_f(self,coords):
        u = self.compose_soln(coords)

        #print(np.shape(u_flat))
        res_separate,res = residual(self.mesh,u ,self.uTest,2,self.sigma,refNorm = self.refNorm)
        self.all_res.append(res_separate)
        return [res]

    """ All the remaining optimizer interfaces are left blank """
    def con_cieq(self,coords):
        return []

    def con_ceq(self,coords):
        return []
    
    def con_dceq(self,coords):
        return []
    
    def obj_df(self,coords):
        return []

    def con_dcieq(self,coords):
        return []
    
    # Post-process the optimization solution
    def solnPost(self,output):
        coords = output[0]
        print('coords = ['+', '.join([str(c) for c in coords])+']')
        u_computed = self.compose_soln(coords)

        var_names = ['Density','velocity x','velocity y','pressure']
        
        res_separate,res = residual(self.mesh,u_computed,self.uTest,2,self.sigma,refNorm = self.refNorm)
        print("residual at end", res)     

        plt.figure()
        for var in range(np.shape(u_computed)[1]):
            
            plt.subplot(2,2,var+1)

            cs1 = plt.tricontour(self.nodes[:,0],self.nodes[:,1],u_computed[:,var],20)
            cs2 = plt.tricontour(self.nodes[:,0],self.nodes[:,1],self.uTest[:,var],20,colors = 'red')
            
            h1,_ = cs1.legend_elements()
            h2,_ = cs2.legend_elements()
            plt.legend([h1[0], h2[0]], ['Computed', 'Actual'])
            plt.title('Contour Plot '+var_names[var])

            
        plt.show()

        nodes_lower = np.unique(self.mesh.getMarkCells('lowerwall')[0][0][0])

        for var in range(np.shape(u_computed)[1]):

            plt.scatter(self.nodes[nodes_lower,0],u_computed[nodes_lower,var],label = 'Computed')
            plt.scatter(self.nodes[nodes_lower,0],self.uTest[nodes_lower,var],label='Actual')
            plt.legend()
            plt.title('Surface values '+var_names[var])
            plt.show()