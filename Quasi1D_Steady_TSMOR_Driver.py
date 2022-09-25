# Driver program for Nair Balajewicz' (2019) Transported Snapshot Model Order
# Reduction method applied to the quasi 1D steady C-D nozzle flow problem

from distutils.log import error
import numpy as np
import os
from optparse import OptionParser
import matplotlib.pylab as plt
plt.rcParams.update({'font.size': 10})
from MyPythonCodes.tools.scipy_tools import scipy_slsqp

from MyPythonCodes.mesh import su2MeshData,UnstructuredMesh, getFileExtUSMD, meshCellFaceProps, polyMeshMetrics

import Quasi1D_Steady_TSMOR as tsmor

from pysph.base.utils import get_particle_array
from pysph.tools.interpolator import Interpolator

parser = OptionParser(usage="usage: %prog -f filename -p pathOut -d debug")
parser.add_option('-p',dest='path',default='.',help='Output path')

parser.add_option('--nx',dest='nfx',default='3',help='No. of Fourier bases for x')
parser.add_option('--ny',dest='nfy',default='2',help='No. of Fourier bases for y')
parser.add_option('--nxy',dest='nfxy',default='0',help='x dependence on y')
parser.add_option('--nyx',dest='nfyx',default='0',help='y dependence on x')
parser.add_option('--ng',dest='ng',default='1',help='g basis function')
parser.add_option('-t',dest='train',default='0', \
    help='Training before validation (if supplied); else validation only')
parser.add_option('--nsig',dest='nsig',default='5',help='Power of sigma values for penalty at inlet')
(options, args) = parser.parse_args()
nfx = int(options.nfx) #No. of basis f-functions in x grid distortion
nfy = int(options.nfy) #No. of basis f-functions in y grid distortion
nfx_y = int(options.nfxy) #No. of basis f-functions in x grid distortion dependent on y
nfy_x = int(options.nfyx) #No. of basis f-functions in y grid distortion dependent on x 
ng = int(options.ng) #No of g basis functions
nsig = int(options.nsig)
train = int(options.train)
sigma_arr = 10**np.arange(1,nsig+1) #Penalty coeff which is increased steadily
# Form the filename for the transport fields' database
transport_db_fn=os.path.join(options.path,'transport_fields_nfx'+str(nfx)+'_nfy'+str(nfy)+'.npz')

""" ===== Offline stage of TSMOR: load snapshots and compute transports ==== """

print('\n'+'-'*80+'\nOffline stage: Loading snapshots ...\n'+'-'*80)

#mshFn = "/home/manthangoyal/SU2_data/flow_data/unstruct/bump_unstruct_0.6x" + getFileExtUSMD() #Mesh USM File
mshFn = "/home/manthangoyal/SU2_data/mesh_1x_bump" + getFileExtUSMD() #Mesh USM File
#mshFnFull = os.path.join(os.path.abspath(os.getcwd()), mshFn)
mesh_base = UnstructuredMesh(mshFn) #base mesh is the original mesh without any distortion
mesh_base._readMeshNodes() #read the nodes of the mesh in the base mesh
Nodes = mesh_base.getNodes() #Store the base mesh nodes in Nodes [x,y] and the indices 
                             #of the array form the node indices as well 

#plt.scatter(Nodes[:,0],Nodes[:,1])
print(np.shape(Nodes))
#plt.show()      
parameters_base = meshCellFaceProps.CalcSignedArea2d(mesh_base) # this stores the face area and face normal
print("param_base", parameters_base)
print(np.shape(parameters_base))
print(parameters_base[0])
# Load training database
train_db = np.load(os.path.join(options.path,'training_snaps.npz'))#load the training snapshots
Mus_db = train_db['mus']    #Parameters values of training snapshots
nMus = len(Mus_db)          #No. of training snapshots
u_db = train_db['u_db']     #Training snapshot's flow variables 
dMus0 = Mus_db[1]-Mus_db[0] #Normalization constant for Mus_db
Mus_db_norm = Mus_db/dMus0  #Normalized parameter array of training database

#u_db = u_db[:,:,2:4] #Choose the flow variables to be used for calculation [snapshot , grid points , flow variables ]

u_db_normalised = u_db/np.amax(np.amax(u_db,axis=1,keepdims=True),axis=0,keepdims=True)# Normalised flo field using the max of each variable across all the snapshots
u_db_normalised = u_db_normalised[:,:,6:7]
print("max", np.shape(u_db))
nodes_lower = np.unique(mesh_base.getMarkCells('lowerwall')[0][0][0]) #Nodes on the lower wall
coor_lower = mesh_base.getNodes()[nodes_lower]#Co-ordinates of lower wall's nodes 

print(np.shape(u_db))
plt.tricontour(Nodes[:,0],Nodes[:,1],u_db_normalised[0,:,0],20) # choose 20 contour levels, just to show how good its interpolation is
plt.tricontour(Nodes[:,0],Nodes[:,1],u_db_normalised[1,:,0],20,colors='red')
#plt.tricontour(Nodes[:,0],Nodes[:,1],u_db_normalised[2,:,0],20,colors='black') 

plt.colorbar()
plt.show()

u_uniform = np.ones((np.shape(Nodes)[0],1))
h = 4 * np.max(np.diff(Nodes[:,0],axis=0))
m = h**2
pa = get_particle_array(name = "myprop",x=Nodes[:,0],y=Nodes[:,1],\
    density=u_uniform[:,0],h = 1.3*(Nodes[0,0]-Nodes[1,0]),
            m=m)

interp = Interpolator([pa], x=coor_lower[::,0],y=coor_lower[::,1]-0.2,\
    method='shepard')
        
u_extrap = interp.interpolate('density')
print(u_extrap)

# meshx,meshy = np.reshape(Nodes[:,0],(128,256)),\
#     np.reshape(Nodes[:,1],(128,256))
if train==1:
    # Pre-allocate array of grid-distortion coefficients for each snapshot
    trnsprt_Csx = np.zeros((nfx*ng,nMus))
    trnsprt_Csy = np.zeros((nfy*ng,nMus))
    error_distortion = np.zeros(nMus)
    # Loop over snapshots to calculate their grid-distortion field coefficients
    for iRef in range(nMus):
    #for iRef in range(1):
        print('\nOffline stage: Constructing transport field for sampled ' \
            +'snapshot #'+str(iRef+1)+' (mu = '+str(Mus_db[iRef])+') ...')
        # Indices of 2 nearest neighbours for the current reference snapshot in
        # the training database (by parameter value)
        iNbs = np.sort(np.argpartition(np.abs(Mus_db - Mus_db[iRef]),2)[1:3])
        # Create optimization project defining offline TSMOR problem; note that
        # a) We only focus on the first component of the snapshots, and
        # b) We normalize the parameter values for better performance


        project_off=tsmor.Project_TSMOR_Offline(u_db,\
        Mus_db_norm,iRef,iNbs,mesh_base,nfx,nfy,nfx_y,nfy_x,ng=ng)

        # Form the initial guess of solution
        if iRef == 0:   #First snapshot
            coeffsx0 = np.zeros((nfx*ng))   #Nothing better than zeros
            coeffsy0 = np.zeros((nfy*ng))   #Nothing better than zeros
            coeffsx0 = [-0.0142, 0.0413, -0.0377, 0.0790, -0.0207]
            coeffsy0 = [-0.0060, 0.0825, -0.0384]
            #use when both density and vel x are used 
            #coeffsx0 = [-0.0015, 0.0253, -0.0456, 0.0776, -0.0254]
            #coeffsy0 = [-0.0044, 0.0737, -0.0427]
        else:
            coeffsx0 = trnsprt_Csx[:,iRef-1]  #Use previous solution
            coeffsy0 = trnsprt_Csy[:,iRef-1]  #Use previous solution
        # Run the optimization and obtain the output

        coeffs0 = np.concatenate((coeffsx0,coeffsy0)) #array of coefficients
        
        output = scipy_slsqp(project_off,x0=coeffs0,its=100,accu=1e-5, \
            eps=1e-4,disp=1)
        # Grid-distortion coefficients for current snapshot is the solution of
        # the optimization process that is returned as the first entry of
        # 'output'; store it in the overall arRepresentationray
        trnsprt_Csx[:,iRef] = output[0][:nfx*ng]
        trnsprt_Csy[:,iRef] = output[0][nfx*ng:]
        error_distortion[iRef] = output[1]
        project_off.solnPost(output)
    #endfor iRef in range(nMus)   #Done addressing all training snapshots
    #np.savetxt()
    print(error_distortion)
    np.savez(transport_db_fn,mus=Mus_db,Cx=trnsprt_Csx,Cy = trnsprt_Csy,\
        nfx_y = nfx_y,nfy_x = nfy_x,ng=ng)
else:
    print('\nOffline stage: Loading pre-computed transport fields ...')
    trnsprt_db = np.load(transport_db_fn)
    trnsprt_Csx = trnsprt_db['Cx']
    trnsprt_Csy = trnsprt_db['Cy']

print('\n'+'-'*80+'\nEnd of offline stage\n'+'-'*80+'\n')
print('exit statement at line 100 of tsmor driver')

""" = Online stage of TSMOR: predict unsampled snapshots from sampled data = """
print('\n'+'-'*80+'\nOnline stage: Loading testing snapshots ...\n'+'-'*80)
# Load testing database
test_db = np.load(os.path.join(options.path,'testing_snaps.npz'))
Mus_test = test_db['mus']   #Parameter values for testing snapshots
nMus_test = len(Mus_test)   #No. of testing snapshots
u_test = test_db['u_db']    #Testing snapshot's flow variables
Mus_test_norm = Mus_test/dMus0  #Normalized parameter array of testing database 
                                #using normalization factor of training database
u_test = np.concatenate((u_test[:,:,2:5],u_test[:,:,6:7]),axis = 2) # density, x-vel, y-vel, pressure
# Loop over testing database parameters to predict their flow fields
for iTest in range(len(Mus_test)):
    print('\nOnline stage: Prediction at testing snapshot #'+str(iTest+1) \
        +' of '+str(nMus_test)+' (mu = '+str(Mus_test[iTest])+') ...')
    # Indices of 2 nearest neighbours for the current test case (by parameter
    # value) that serve as the reference snapshots to be transported to form the
    # basis set for predicting the current case
    iRefs = np.sort(np.argpartition(np.abs(Mus_db - Mus_test[iTest]),1)[:2])
    assert np.all(np.diff(iRefs)==1),'iRefs array should be contiguous sequence'
    # Create optimization project defining online TSMOR problem; note that
    # normalized parameter of test case is being sent, to be compatible with
    # normalized parameters of training database that were used to derive the
    # grid distortion coefficients in the offline stage
    
    print(iRefs, np.shape(u_db))
    uRefs = np.concatenate((u_db[iRefs[0]:iRefs[-1]+1,:,2:5],u_db[iRefs[0]:iRefs[-1]+1,:,6:7]),\
        axis = 2) # density, x-vel, y-vel, pressure
    dMuRefs = Mus_test_norm[iTest]-Mus_db_norm[iRefs] #parameter difference in test and ref 
    coords0 = (1.0/abs(dMuRefs))/sum(1.0/abs(dMuRefs))#initial weights assigned 
    project_on = tsmor.Project_TSMOR_Online(u_test[iTest,:,:], uRefs, \
        Mus_db_norm[iRefs],trnsprt_Csx[:,iRefs[0]:iRefs[-1]+1],trnsprt_Csy\
            [:,iRefs[0]:iRefs[-1]+1],mesh_base,Mus_test_norm[iTest],coords0,\
                nfx_y,nfy_x,ng=ng,sigma = 10000)

    # Run the optimization and obtain the output
    output = scipy_slsqp(project_on,x0=project_on.coords0,its=100, \
        accu=1e-12,eps=1e-10,disp=1)
    # Post-process the optimal solution (see it and compare with known solution)
    project_on.solnPost(output)
print('\n'+'-'*80+'\nEnd of online stage\n'+'-'*80+'\n')
plt.show()
