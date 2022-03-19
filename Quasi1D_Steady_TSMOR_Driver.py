# Driver program for Nair Balajewicz' (2019) Transported Snapshot Model Order
# Reduction method applied to the quasi 1D steady C-D nozzle flow problem

import numpy as np
import os
from optparse import OptionParser
import matplotlib.pylab as plt

from MyPythonCodes.tools.scipy_tools import scipy_slsqp

from MyPythonCodes.mesh import su2MeshData,UnstructuredMesh, getFileExtUSMD, meshCellFaceProps, polyMeshMetrics

import Quasi1D_Steady_TSMOR as tsmor

parser = OptionParser(usage="usage: %prog -f filename -p pathOut -d debug")
parser.add_option('-p',dest='path',default='.',help='Output path')

parser.add_option('--nx',dest='nfx',default='3',help='No. of Fourier bases for x')
parser.add_option('--ny',dest='nfy',default='3',help='No. of Fourier bases for y')

parser.add_option('-t',dest='train',action="store_true",default=True, \
    help='Training before validation (if supplied); else validation only')

(options, args) = parser.parse_args()
nfx = int(options.nfx) #No. of basis f-functions in x grid distortion
nfy = int(options.nfy) #No. of basis f-functions in y grid distortion

# Form the filename for the transport fields' database
transport_db_fn=os.path.join(options.path,'transport_fields_nfx'+str(nfx)+'_nfy'+str(nfy)+'.npz')

""" ===== Offline stage of TSMOR: load snapshots and compute transports ==== """

print('\n'+'-'*80+'\nOffline stage: Loading snapshots ...\n'+'-'*80)


#mesh_file = "home/manthangoyal/SU2_data/mesh_1x_bump.su2"  #File where mesh is located
#mesh = su2MeshData.SU2Mesh()                   #Object of su2mesh class
#mesh.read(mesh_file)                           #mesh of the ebump is read
#points = mesh.getNodes()                       #2D array of points along rows and their
#                                               #co ordinates along columns

# load the mesh in USM format
mshFn = "/home/manthangoyal/SU2_data/mesh_1x_bump" + getFileExtUSMD() #Mesh USM File
#mshFnFull = os.path.join(os.path.abspath(os.getcwd()), mshFn)
mesh_base = UnstructuredMesh(mshFn)
mesh_base._readMeshNodes()
Nodes = mesh_base.getNodes()
#plt.scatter(Nodes[:,0],Nodes[:,1])
print(np.shape(Nodes))
#plt.show()      
parameters_base = meshCellFaceProps.CalcSignedArea2d(mesh_base)
print("param_base", parameters_base)
print(np.shape(parameters_base))
print(parameters_base[0])
# Load training database
train_db = np.load(os.path.join(options.path,'training_snaps.npz'))
Mus_db = train_db['mus']    #Parameters values of training snapshots
nMus = len(Mus_db)          #No. of training snapshots
u_db = train_db['u_db']     #Training snapshot's flow variables 
dMus0 = Mus_db[1]-Mus_db[0] #Normalization constant for Mus_db
Mus_db_norm = Mus_db/dMus0  #Normalized parameter array of training database

u_db = u_db[:,:,2:3]/10 #only density
print(np.shape(u_db))
plt.tricontour(Nodes[:,0],Nodes[:,1],u_db[0,:,0],20) # choose 20 contour levels, just to show how good its interpolation is
plt.show()

if options.train:
    # Pre-allocate array of grid-distortion coefficients for each snapshot
    trnsprt_Csx = np.zeros((nfx,nMus))
    trnsprt_Csy = np.zeros((nfy,nMus))

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
        Mus_db_norm,iRef,iNbs,mesh_base,nfx,nfy)
        #project_off.solnPost(np.zeros((nfx+nfy,2)))
        # Form the initial guess of solution
        if iRef == 0:   #First snapshot
            coeffsx0 = np.zeros((nfx))   #Nothing better than zeros
            coeffsy0 = np.zeros((nfy))   #Nothing better than zeros
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
        trnsprt_Csx[:,iRef] = output[0][:nfx]
        trnsprt_Csy[:,iRef] = output[0][:nfy]
        #print("u  ",u_db[:4,:,iRef])
        #print("u  ",u_db[:4,:,iRef+1])
        #print("u normalised  ",u_db_norm[:4,:,iRef+1])

        #print('normalised ',normalised)
        #print("trns coeff  ",trnsprt_Cs[:,iRef])
        # Post-process the optimal solution (see it)
        #project_off.printu()
        project_off.solnPost(output)
    #endfor iRef in range(nMus)   #Done addressing all training snapshots
    np.savez(transport_db_fn,mus=Mus_db,Cx=trnsprt_Csx,Cy = trnsprt_Csy)
else:
    print('\nOffline stage: Loading pre-computed transport fields ...')
    trnsprt_db = np.load(transport_db_fn)
    trnsprt_Cs = trnsprt_db['C']

print('\n'+'-'*80+'\nEnd of offline stage\n'+'-'*80+'\n')
print('exit statement at line 100 of tsmor driver')
exit()

""" = Online stage of TSMOR: predict unsampled snapshots from sampled data = """
print('\n'+'-'*80+'\nOnline stage: Loading testing snapshots ...\n'+'-'*80)
Am = A_db[0,0];     xi = xarr[0];   xf = xarr[-1]
# Load testing database
test_db = np.load(os.path.join(options.path,'testing_snapshots.npz'))
Mus_test = test_db['mus']   #Parameter values for testing snapshots
nMus_test = len(Mus_test)   #No. of testing snapshots
xarr_test = test_db['xarr'] #x-grid on which testing snapshots are defined
u_test = test_db['u_db']    #Testing snapshot's flow variables
Mus_test_norm = Mus_test/dMus0  #Normalized parameter array of testing database 
                                #using normalization factor of training database
prblm_setups = test_db['prblm_setups'] #Problem setup data structures

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
    project_on = tsmor.Project_TSMOR_Online(u_db[:,:,iRefs[0]:iRefs[-1]+1], \
        Mus_db_norm[iRefs],trnsprt_Cs[:,iRefs[0]:iRefs[-1]+1],xarr_test, \
        Mus_test_norm[iTest],prblm_setups[:,iTest])
    # Run the optimization and obtain the output
    output = scipy_slsqp(project_on,x0=project_on.coords0,its=100, \
        accu=1e-12,eps=1e-10,disp=1)
    # Post-process the optimal solution (see it and compare with known solution)
    project_on.solnPost(output,u_test[:,:,iTest])
print('\n'+'-'*80+'\nEnd of online stage\n'+'-'*80+'\n')
plt.show()
