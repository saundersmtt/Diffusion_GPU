import MDAnalysis as mda
from MDAnalysis import transformations
import numpy as np
import math
import sys
import argparse
import warnings
import time
from tqdm import tqdm
import logging
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import linregress




def main():

    parser=argparse.ArgumentParser(description="",epilog="e.g.  ")
    parser.add_argument("Group_1", type=str, help="Group of atoms to compute diffusion coefficient for, use select language to choose them")
    parser.add_argument("-s", "--top", type=str, help="Topology file")
    parser.add_argument("-f", "--traj", type=str, help="Trajectory file")
    parser.add_argument("-o", "--output", type=str, help="output data")
    parser.add_argument("-d","--dim",type=str,help="Dimension X,Y, or Z to divide the box into slices. If not provided,"
    "box is not sliced",default="None")
    parser.add_argument("-n","--nslices",help="Number of slices to divide the box into if a dimension is given",default=1)
    parser.add_argument("-t","--lagtime",help="Length of the MSD to use for fit. Trajectory will be split into chunks of this length, and each particle in the chosen group will have their MSD calculated for this amount of time before resetting the starting position.",default=100)    
    parser.add_argument("-v","--verbose",help="Enables helpful logging information",action="store_true")
    parser.add_argument("--debug",help="Enables helpful debug information, including built-in debugging for MDA tools",action="store_true")
    parser.add_argument("--num",help="Number of frames to run on")
    parser.add_argument("--fit_start", help="Lagtime index to start linear fit", type=int, default=10)
    parser.add_argument("--fit_end", help="Lagtime index to end linear fit", type=int, default=50)
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)  #Added this so I don't have to comment out my testing print lines this time
    if args.debug:
        logging.basicConfig(level=logging.DEBUG) #This logging level enables mdanalysis debug notes as well, which is useful
    
    if args.top is None:
        top_file="topology.tpr"
    else:
        top_file=args.top
    
    if args.traj is None:
        traj_file="traj.trr"
    else:
        traj_file=args.traj.split() #Can be a single trajectory, or a list of trr files to be read continuously
    
    if args.output is None:
        output_name="output"
    else:
        output_name=args.output
    
    # Initializing the MD universe
    logging.info("Initializing the MDA universe...")
    md=mda.Universe(top_file,traj_file,continuous=True) #Continuous flag ensures no duplicate frames when reading a list of trr's
    logging.info(mda.__version__)
    
    if args.num is None:
        num_frames=len(md.trajectory)
    else:
        num_frames=int(args.num)

    search_str="(("+args.Group_1+"))"
    group1=md.select_atoms(args.Group_1)
    n_atoms = len(group1)
    pos_list = []
    
    logging.info(f"Stacking {num_frames} frames...")
    for i, ts in enumerate(tqdm(md.trajectory, total=num_frames)):
        if i >= num_frames:
            break
        pos_list.append(group1.positions.copy())
    
    positions = np.stack(pos_list, axis=0)  # (n_frames, n_atoms, 3)
    positions_tf = tf.convert_to_tensor(positions, dtype=tf.float32)
   # do_slicing = (args.dim.upper() in ("X", "Y", "Z")) and (args.nslices > 1)
    
    # Set up lagtimes
    max_lag = int(args.lagtime)
    lagtimes = np.arange(1, max_lag + 1)
    
    all_msd = []
    
    logging.info("Computing per-atom MSDs vs lagtime...")
    
    for lag in tqdm(lagtimes):
        disp = positions_tf[lag:] - positions_tf[:-lag]  # (n_frames - lag, n_atoms, 3)
        sq_disp = tf.reduce_sum(tf.square(disp), axis=-1)  # (n_frames - lag, n_atoms)
        msd_lag = tf.reduce_mean(sq_disp, axis=0)  # (n_atoms,)
        all_msd.append(msd_lag.numpy())
    
    all_msd = np.stack(all_msd, axis=0)  # (n_lagtimes, n_atoms)
    
    # Fit per-atom MSD curves
    from scipy.stats import linregress
    
    fit_start = args.fit_start  
    fit_end = args.fit_end    
    
    diffusion_coeffs = []
    
    for i in range(all_msd.shape[1]):  # loop over atoms
        slope, intercept, _, _, _ = linregress(lagtimes[fit_start:fit_end], all_msd[fit_start:fit_end, i])
        D = slope / 6  # assuming 3D diffusion
        diffusion_coeffs.append(D)
    
    diffusion_coeffs = np.array(diffusion_coeffs)
    
    # Save results
    np.savetxt(f"{output_name}_diffusion_coefficients.dat", diffusion_coeffs)
    
    # Optionally plot histogram
    import matplotlib.pyplot as plt
    plt.hist(diffusion_coeffs, bins=30, alpha=0.7)
    plt.xlabel("Diffusion Coefficient ($\\mathrm{\\AA^2/frame}$)")
    plt.ylabel("Count")
    plt.title("Distribution of Diffusion Coefficients")
    plt.grid(True)
    plt.savefig(f"{output_name}_diffusion_histogram.png")
    plt.close()
    
    logging.info(f"Average Diffusion Coefficient: {np.mean(diffusion_coeffs):.5f} Å²/frame")



