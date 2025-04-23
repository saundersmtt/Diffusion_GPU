import numpy as np
import sys
import argparse
from tqdm import tqdm
import logging     #Enables us to use a debug flag to print some useful information
from numba import jit,objmode
import os
import pickle



def count_frames(filename, header="ITEM: ATOMS"):
    frame_count = 0
    with open(filename, 'r') as f:
        for line in f:
            if header in line:
                frame_count += 1
    return frame_count

@jit(nopython=True)
def put_atom_in_box(x,box,minvec):
    "Puts atom in box using box vectors"
    x -= minvec
    for d in range(3):
        boxVectorShift = np.floor(x[d] * 1/box[d]) 
        p = x[d] - (boxVectorShift * box[d])
        x[d] = p
    return x + minvec

@jit(nopython=True)
def getbin(pos,width,minpos):
        "Computes the histogram bin index."
        x = pos - minpos 
        n_bin = int(np.floor((x) / width))
        center = (n_bin*width + width/2) + minpos
        return n_bin,center

@jit(nopython=True)
def read_atom_data(atoms_data,minvec,maxvec,box,selection,frame):
    n = 0
    for i in range(atoms_data.shape[0]):
        if atoms_data[i, 1] == selection:
            n += 1

    positions = np.empty((n, 5), dtype=atoms_data.dtype)

    idx = 0
    for i in range(atoms_data.shape[0]):
        if int(atoms_data[i, 1]) == selection:
            positions[idx, 0] = atoms_data[i, 0]
            positions[idx, 1] = atoms_data[i, 2]
            positions[idx, 2] = atoms_data[i, 3]
            positions[idx, 3] = atoms_data[i, 4]
            positions[idx, 4] = frame
            idx += 1

    return positions




def main():
    parser=argparse.ArgumentParser(description="",epilog="e.g.  ")
    parser.add_argument("-v","--verbose",help="Enables helpful logging information",action="store_true")
    parser.add_argument("--debug",help="Enables helpful debug information, including built-in debugging for MDA tools",action="store_true")
    parser.add_argument("-f", "--traj", type=str, help="Trajectory file")
    parser.add_argument("-o", "--output", type=str, help="output data")
    parser.add_argument("-s","--selection",type=int,help="Atom type for diffusion calculation",required=True)

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)  #Added this so I don't have to comment out my testing print lines
    if args.debug:
        logging.basicConfig(level=logging.DEBUG) 

    trajectories=args.traj.strip().split()
    qframes = []
    dipframes = []
    trajectory = args.traj
    delimiter = b"ITEM: TIMESTEP"
    charge=0.
    diphistogram = np.asarray((False))
    ATOMS=False
    of=None
    of_variance=None
    positions_data=np.array([False])
    atoms_data=np.array([False])
    for trajectory in trajectories:
        if positions_data.any():
            pickle.dump(positions_data,of)
            positions_data = np.array([False])
        nframes = count_frames(trajectory)
        pbar = tqdm(total=nframes,unit='frames')
        trajname = os.path.basename(trajectory)
        dipolelist = []
        outfilename=args.output+"_"+str(trajname)
        if of:
            of.close()
        of = open(outfilename+"_output.pkl", 'ab')
        with open(trajectory, "r") as f:
            for line in f:
                line = line.strip().split()
                if len(line) == 0:
                    continue
                if line[0] == "ITEM:":
                    if line[1] == "NUMBER":
                        natoms = int(next(f).split()[0])
                        continue
                    elif line[1] == "TIMESTEP":
                        timestep = int(next(f).split()[0])
                        #if positions_data.any():
                        #    pickle.dump(positions_data,of)
                        #    positions_data = np.array([False])
                        dims = []
                        pbar.update(1)
                        continue
                    elif line[1] == "BOX":
                        for _ in range(3):
                            line = next(f).split()
                            dims.append(line)
                            logging.info(dims)
                        dims = np.asarray(dims,dtype=np.float64)
                        continue
                    elif line[1] == "ATOMS":
                        maxvec=dims[:3,1].flatten()
                        minvec=dims[:3,0].flatten()
                        box = maxvec - minvec
                        ATOMS=True
                if ATOMS:
                    logging.debug(f"Reading atoms!")
                    atoms_data = np.loadtxt(f, max_rows=natoms, dtype=np.float64)
                    #ITEM: ATOMS id type xu yu zu vx vy vz q
    #                print(np.shape(atoms_data))
#                    np.vstack((positions_data,read_atom_data(atoms_data,minvec,maxvec,box,args.selection)))
                    selected = read_atom_data(atoms_data,minvec,maxvec,box,args.selection,timestep)
                    logging.debug(np.shape(selected))
                    logging.debug(selected)
    #                for line in selected:
    #                    print(f"{line}")
                    if positions_data.any():
                        positions_data = np.vstack((positions_data,selected))
                    else:
                        positions_data = selected
#                    print(np.shape(positions_data))
                    logging.debug(f"Finished frame {timestep}!")
                    del atoms_data
                    atoms_data = np.array([False])
                    ATOMS=False
    
#    with open('my_array.pkl', 'wb') as f:
#    pickle.dump(arr, f)
    pickle.dump(positions_data,of)


if __name__ == "__main__":
    main()
