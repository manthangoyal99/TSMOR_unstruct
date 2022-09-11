from ast import keyword
import os 
import numpy as np
import matplotlib.pylab as plt
from optparse import OptionParser
from MyPythonCodes.mesh import su2MeshData, su2Mesh2USM, UnstructuredMeshData, \
    UnstructuredMesh, getFileExtUSMD, MeshCellFaceProps

def save_mesh_data():

    parser = OptionParser(usage="usage: %prog -f filename -p pathOut -d debug")
    parser.add_option('-p',dest='path',default='.',help='Output path')
    (options,args) = parser.parse_args()
    #outfile = os.path.join(options.path,'testing_snaps.npz')
    outfile = os.path.join(options.path,'training_snaps.npz')
    #mesh_file = "/home/manthangoyal/SU2_data/mesh_1x_bump.su2"  #File where struct mesh is located
    mesh_file = "/home/manthangoyal/SU2_data/flow_data/unstruct/bump_unstruct_0.6x.su2"  #File where unstruct mesh is located
    #mesh_file = "/home/manthangoyal/SU2_data/MESHPROPTEST.su2"  
    #mesh = su2MeshData.SU2Mesh()                   #Object of su2mesh class
    #mesh.read(mesh_file)
    keywords = dict(keyword1 = 'debug')
    mesh = su2Mesh2USM.SU2Mesh2USM(mesh_file,**keywords)
    mesh.toUSM()
    mshFn = "/home/manthangoyal/SU2_data/flow_data/unstruct/bump_unstruct_0.6x" + getFileExtUSMD()
    mshFnFull = mshFn

    msh = UnstructuredMesh(mshFn)
    msh.updtMetaMeshMrkrNm(meshFile=mshFnFull)
    msh.writeMeta()

    mcfp = MeshCellFaceProps(mshFn)
    mcfp.calc()
    
    machs = np.array([.7,.73,.76,.79,.82,.85])
    #machs = np.array([.72,.74,.75,.81])
    flow_data = []

    for mach in machs:
        """for struct"""
        #flow_file = "/home/manthangoyal/SU2_data/flow_data/bump-"+str(mach)+"/restart_flow.dat"
        """for unstruct"""
        flow_file = "/home/manthangoyal/SU2_data/flow_data/unstruct/bump-"+str(mach)+"/restart_flow.dat"
        flow_read = su2MeshData.SU2MeshData()
        flow_read.readDataOnly(flow_file)
        flow_data.append(flow_read.data)
    
    flow_data_numpy = np.array(flow_data)
    """
    x -> no. of snapshots
    y -> points
    z-> flow variables with first 2 as x and y
    """
    #print(flow_data_numpy[0,:20,2])
    np.savez(outfile,mus = machs, u_db = flow_data_numpy)

if __name__=="__main__":
    save_mesh_data()
