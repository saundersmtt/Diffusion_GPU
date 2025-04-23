import numpy as np
import argparse
import logging
from tqdm import tqdm
import tensorflow as tf
from scipy.stats import linregress
import matplotlib.pyplot as plt
import pickle

#-------------------------------------------------------------------------------
@tf.function
def compute_all_msd(positions, lagtimes):
    """
    Compute MSD for all lag times in one GPU-fused graph call.
    positions: tensor, shape (n_frames, n_atoms, 3)
    lagtimes: 1D int32 tensor of lag indices [1, 2, ..., max_lag]
    returns: tensor shape (n_lagtimes, n_atoms)
    """
    def msd_for_lag(lag):
        # Position differences for this lag
        disp = positions[lag:] - positions[:-lag]            # (n_frames-lag, n_atoms, 3)
        sq   = tf.reduce_sum(tf.square(disp), axis=-1)       # (n_frames-lag, n_atoms)
        return tf.reduce_mean(sq, axis=0)                    # (n_atoms,)

    # Map the msd_for_lag fn over all lagtimes in one graph
    return tf.map_fn(msd_for_lag, lagtimes, dtype=positions.dtype)
#-------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute per-atom MSDs and diffusion coefficients using TF/GPU.")
    parser.add_argument("-f", "--trajectory", nargs='+', default=["traj.pkl"],
                        help="Trajectory files (space-separated list) or single file. Pickled array of positions after selection")
    parser.add_argument("-o", "--output", type=str, default="output",
                        help="Base name for output files.")
    parser.add_argument("-t", "--lagtime", type=int, default=100,
                        help="Maximum lag (in frames) for MSD calculation.")
    parser.add_argument("--fit_start", type=int, default=10,
                        help="Lag index to start linear fit.")
    parser.add_argument("--fit_end",   type=int, default=50,
                        help="Lag index to end linear fit.")
    parser.add_argument("--num", type=int,
                        help="Number of frames to process (default: all).")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable INFO logging.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG logging.")
    args = parser.parse_args()

    # Logging setup
    level = (logging.DEBUG if args.debug else
             logging.INFO if args.verbose else
             logging.WARNING)
    logging.basicConfig(level=level)

    trajectory_files = args.trajectory
    trajectory = np.array([False])
    for file in args.trajectory:
        with open(file,"rb") as f:
            trajectories.append(pickle.load(f))
    trajectory=np.vstack(trajectories)
        # trajectory: shape (nframes*natoms, 4)  with columns [id, x, y, z]
    ids    = trajectory[:, 0].astype(int)
    coords = trajectory[:, 1:]               # shape (nframes*natoms, 3)
    
    # figure out how many atoms & frames
    unique_ids = np.unique(ids)
    natoms     = unique_ids.size
    nframes    = trajectory.shape[0] // natoms
    print(natoms,nframes,np.shape(trajectory))
    assert nframes * natoms == trajectory.shape[0]
    
    # build a lookup from atom ID → row index in positions
    id_to_index = { atom_id: idx for idx, atom_id in enumerate(unique_ids) }
    
    # allocate output: (natoms, nframes, 3)
    positions = np.empty((natoms, nframes, 3), dtype=coords.dtype)
    
    # scatter each frame’s block into positions
    for frame in range(nframes):
        start = frame * natoms
        end   = start + natoms
        block = trajectory[start:end]        # shape (natoms, 4)
        block_ids    = block[:, 0].astype(int)
        block_coords = block[:, 1:]          # shape (natoms, 3)
    
        # map IDs to the 0…natoms-1 row indices
        indices = [id_to_index[a] for a in block_ids]
    
        # place x,y,z into positions[:, frame, :]
        positions[indices, frame, :] = block_coords
    
    # now positions.shape == (natoms, nframes, 3)

    # Move to TensorFlow GPU
    positions_tf = tf.convert_to_tensor(positions, dtype=tf.float32)

    # Prepare lag times
    max_lag = args.lagtime
    lagtimes = np.arange(1, max_lag + 1)
    lagtimes_tf = tf.constant(lagtimes, dtype=tf.int32)

    # Compute all MSDs in one GPU call
    logging.info("Computing all MSDs on GPU...")
    all_msd_tf = compute_all_msd(positions_tf, lagtimes_tf)
    all_msd    = all_msd_tf.numpy()  # shape (n_lagtimes, n_atoms)

    # Save per-atom MSD curves
    logging.info("Writing per-atom MSD curves...")
    for i in range(n_atoms):
        msd_curve = all_msd[:, i]
        out = np.column_stack((lagtimes, msd_curve))
        fn  = f"{args.output}_msd_atom_{i:04d}.dat"
        np.savetxt(fn, out, header="lagtime\tmsd", comments="")

    # Fit per-atom diffusion coefficients
    fit_start, fit_end = args.fit_start, args.fit_end
    diffusion_coeffs = []
    for i in range(n_atoms):
        slope, intercept, _, _, _ = linregress(
            lagtimes[fit_start:fit_end],
            all_msd[fit_start:fit_end, i]
        )
        D = slope / 6.0
        diffusion_coeffs.append(D)
    diffusion_coeffs = np.array(diffusion_coeffs)

    # Save diffusion coefficients
    diff_fn = f"{args.output}_diffusion_coeffs.dat"
    np.savetxt(diff_fn, diffusion_coeffs,
               header=f"# per-atom D (Å^2/frame) for {n_atoms} atoms", comments="")

    # Plot histogram of D
    plt.hist(diffusion_coeffs, bins=30, alpha=0.7)
    plt.xlabel("Diffusion Coefficient (Å²/frame)")
    plt.ylabel("Count")
    plt.title("Distribution of Per-Atom Diffusion Coefficients")
    plt.grid(True)
    plt.savefig(f"{args.output}_diffusion_histogram.png")
    plt.close()

    logging.info(f"Average D: {diffusion_coeffs.mean():.5f} Å²/frame")


if __name__ == "__main__":
    main()

