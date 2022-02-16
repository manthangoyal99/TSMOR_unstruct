import os 
import numpy as np
import matplotlib.pylab as plt
from optparse import OptionParser
from MyPythonCodes.mesh import su2MeshData

def save_mesh_data():

    parser = OptionParser(usage="usage: %prog -f filename -p pathOut -d debug")
    parser.add_option('-p',dest='path',default='.',help='Output path')
    (ptions,args) = parser.parse_args()
    outfile = os.path.join(options.path,'training_snaps.npz')

    machs = np.array([.7,.73,.76,.8,82,85])

    flow_data = []

    for mach in machs:
        flow_file = "/home/manthangoyal/SU2_data/flow_data"
        flow_read = su2MeshData.SU2MeshData()
        flow_read.readDataOnly(flow_f)
        flow_data.append(flow_read.data)
    
    flow_data_numpy = np.array(flow_data)
    np.savez(outfile,mus = machs, u_db = flow_data_numpy)

if __name__as"__main__":
    save_mesh_data()
