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




def main():

    parser=argparse.ArgumentParser(description="",epilog="e.g.  ")
    parser.add_argument("Group_1", type=str, help="Group of atoms to compute diffusion coefficient for, use select language to choose them")
    parser.add_argument("-s", "--top", type=str, help="Topology file")
    parser.add_argument("-f", "--traj", type=str, help="Trajectory file")
    parser.add_argument("-o", "--output", type=str, help="output data")
    parser.add_argument("-d","--dim",type=str,help="Dimension X,Y, or Z to divide the box into slices. If not provided,
    box is not sliced",default="None")
    parser.add_argument("-n","--nslices",help="Number of slices to divide the box into if a dimension is given",default=1)
    parser.add_argument("-t","--lagtime",help="Length of the MSD to use for fit. Trajectory will be split into chunks of this length, and each particle in the chosen group will have their MSD calculated for this amount of time before resetting the starting position.",default=100)    
    parser.add_argument("-v","--verbose",help="Enables helpful logging information",action="store_true")
    parser.add_argument("--debug",help="Enables helpful debug information, including built-in debugging for MDA tools",action="store_true")
    parser.add_argument("--num",help="Number of frames to run on")
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


