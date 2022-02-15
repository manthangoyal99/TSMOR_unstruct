# Driver program for Nair Balajewicz' (2019) Transported Snapshot Model Order
# Reduction method applied to the quasi 1D steady C-D nozzle flow problem

import numpy as np
import os
from optparse import OptionParser
import matplotlib.pylab as plt

from MyPythonCodes.tools.scipy_tools import scipy_slsqp

import Quasi1D_Steady_TSMOR as tsmor

parser = OptionParser(usage="usage: %prog -f filename -p pathOut -d debug")
parser.add_option('-p',dest='path',default='.',help='Output path')
parser.add_option('-n',dest='nf',default='3',help='No. of Fourier bases')
parser.add_option('--nor',dest = 'normalised',action = "store_true",default = False, \
    help='Are normalised variables needed')
parser.add_option('-t',dest='train',action="store_true",default=False, \
    help='Training before validation (if supplied); else validation only')
(options, args) = parser.parse_args()
nf = int(options.nf) #No. of basis f-functions in grid distortion
normalised = options.normalised #if normalised variables are there or not
# Form the filename for the transport fields' database
transport_db_fn=os.path.join(options.path,'transport_fields_nf'+str(nf)+'.npz')

""" ===== Offline stage of TSMOR: load snapshots and compute transports ==== """

print('\n'+'-'*80+'\nOffline stage: Loading snapshots ...\n'+'-'*80)

def nor_flow_variables(u_db):
    """
    Calculates the normalised flow variables
    """
    u_db_inlet = u_db[0,:,:]
    print("inlet variables " ,np.shape(u_db_inlet))
    return u_db/u_db_inlet,u_db_inlet

def grad_flow_variables(u_db,xarr):

    u_db_grad = np.zeros_like(u_db)

    xarr_stacked2d = np.vstack((xarr[2:]-xarr[:-2],xarr[2:]-xarr[:-2],xarr[2:]-xarr[:-2]))
    xarr_stacked3d = np.stack((xarr_stacked2d.T,xarr_stacked2d.T,xarr_stacked2d.T,xarr_stacked2d.T),axis = 2)
    u_db_grad[1:-1,:,:] =   (u_db[2:,:,:]-u_db[:-2,:,:])/xarr_stacked3d
    u_db_grad[0,:,:] = (u_db[1,:,:]-u_db[0,:,:])/(xarr[1]-xarr[0])
    u_db_grad[-1,:,:] = (u_db[-1,:,:]-u_db[-2,:,:])/(xarr[-1]-xarr[-2])

    return u_db_grad

# Load training database
train_db = np.load(os.path.join(options.path,'training_snapshots.npz'))
Mus_db = train_db['mus']    #Parameters values of training snapshots
nMus = len(Mus_db)          #No. of training snapshots
xarr = train_db['xarr']     #x-grid on which training snapshots are defined
A_db = train_db['A_db']     #Area distributions for the training snapshots
u_db = train_db['u_db']     #Training snapshot's flow variables
dMus0 = Mus_db[1]-Mus_db[0] #Normalization constant for Mus_db
Mus_db_norm = Mus_db/dMus0  #Normalized parameter array of training database
u_db_norm ,u_db_inlet = nor_flow_variables(u_db) #Normalized flow vaiable matrix of training database

# new variables just having u
u = u_db[:,1,:]/u_db[:,0,:]
u_norm = u_db_norm[:,1,:]/u_db_norm[:,0,:]
u_inlet = u_db_inlet/u_db_inlet[0,:]

u_db_grad = grad_flow_variables(u_db,xarr)

u_db_grad_norm,u_db_grad_inlet = nor_flow_variables(u_db_grad)

#print(np.shape(u_inlet))

if options.train:
    # Pre-allocate array of grid-distortion coefficients for each snapshot
    trnsprt_Cs = np.zeros((nf,nMus))
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



        variable_i = 0 #If using singe variable in the offline case then 0 1 2 determines 
        #which one according to the column number

        all_variables =  True #If using all the variables

        
        #This snippet as written to process for velocity as u_norm is normaised velocity
        #= rho*u/rho
        '''
        if (normalised==True):
            project_off=tsmor.Project_TSMOR_Offline(u_norm.reshape(-1,1,nMus),\
                Mus_db_norm,iRef,iNbs,xarr,normalised,u_inlet,all_variables)

        else:
            project_off=tsmor.Project_TSMOR_Offline(u.reshape(-1,1,nMus),\
                Mus_db_norm,iRef,iNbs,xarr,normalised,np.ones((np.shape(u_db[0,:,:]))),all_variables)
        '''

        
        if(all_variables):
            if (normalised==True):
                project_off=tsmor.Project_TSMOR_Offline(u_db_norm,\
                    Mus_db_norm,iRef,iNbs,xarr,normalised,u_db_inlet,all_variables)

            else:
                project_off=tsmor.Project_TSMOR_Offline(u_db,\
                    Mus_db_norm,iRef,iNbs,xarr,normalised,np.ones((np.shape(u_db[0,:,:]))),all_variables)
                #project_off=tsmor.Project_TSMOR_Offline(u_db_grad,\
                    #Mus_db_norm,iRef,iNbs,xarr,normalised,np.ones((np.shape(u_db[0,:,:]))),all_variables)

        else:

            if (normalised==True):
                #project_off=tsmor.Project_TSMOR_Offline(u_db_norm[:,variable_i,:].reshape(-1,1,nMus),\
                #    Mus_db_norm,iRef,iNbs,xarr,normalised,u_db_inlet,all_variables)
                project_off=tsmor.Project_TSMOR_Offline(u_db_grad_norm[:,variable_i,:].reshape(-1,1,nMus),\
                    Mus_db_norm,iRef,iNbs,xarr,normalised,u_db_grad_inlet,all_variables)
            else:
                #project_off=tsmor.Project_TSMOR_Offline(u_db[:,variable_i,:].reshape(-1,1,nMus),\
                #    Mus_db_norm,iRef,iNbs,xarr,normalised,np.ones((np.shape(u_db[0,:,:]))),all_variables)
                project_off=tsmor.Project_TSMOR_Offline(u_db_grad[:,variable_i,:].reshape(-1,1,nMus),\
                    Mus_db_norm,iRef,iNbs,xarr,normalised,np.ones((np.shape(u_db[0,:,:]))),all_variables)
              
        # Form the initial guess of solution
        if iRef == 0:   #First snapshot
            coeffs0 = np.zeros((nf))   #Nothing better than zeros
        else:
            coeffs0 = trnsprt_Cs[:,iRef-1]  #Use previous solution
        # Run the optimization and obtain the output
        output = scipy_slsqp(project_off,x0=coeffs0,its=100000,accu=1e-12, \
            eps=1e-10,disp=1)
        # Grid-distortion coefficients for current snapshot is the solution of
        # the optimization process that is returned as the first entry of
        # 'output'; store it in the overall arRepresentationray
        trnsprt_Cs[:,iRef] = output[0]
        print("u  ",u_db[:4,:,iRef])
        #print("u  ",u_db[:4,:,iRef+1])
        #print("u normalised  ",u_db_norm[:4,:,iRef+1])

        print('normalised ',normalised)
        print("trns coeff  ",trnsprt_Cs[:,iRef])
        # Post-process the optimal solution (see it)
        project_off.printu()
        project_off.solnPost(output,iRef)
    #endfor iRef in range(nMus)   #Done addressing all training snapshots
    np.savez(transport_db_fn,mus=Mus_db,C=trnsprt_Cs)
else:
    print('\nOffline stage: Loading pre-computed transport fields ...')
    trnsprt_db = np.load(transport_db_fn)
    trnsprt_Cs = trnsprt_db['C']

print('\n'+'-'*80+'\nEnd of offline stage\n'+'-'*80+'\n')
print('exit statement at line 100 of tsmor driver')
#exit()

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
    output = scipy_slsqp(project_on,x0=project_on.coords0,its=100000, \
        accu=1e-12,eps=1e-10,disp=1)
    # Post-process the optimal solution (see it and compare with known solution)
    project_on.solnPost(output,u_test[:,:,iTest])
print('\n'+'-'*80+'\nEnd of online stage\n'+'-'*80+'\n')
plt.show()
