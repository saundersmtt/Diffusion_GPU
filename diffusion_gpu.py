import MDAnalysis as mda
import numpy as np
import argparse
import logging
from tqdm import tqdm
import tensorflow as tf
from scipy.stats import linregress
import matplotlib.pyplot as plt

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
    parser.add_argument("Group_1", type=str,
                        help="Atom selection string for MDAnalysis (e.g. 'resname SOL').")
    parser.add_argument("-s", "--top",  type=str, default="topology.tpr",
                        help="Topology file (TPR, PSF, etc.)")
    parser.add_argument("-f", "--traj", nargs='+', default=["traj.trr"],
                        help="Trajectory files (space-separated list) or single file.")
    parser.add_argument("-o", "--output", type=str, default="output",
                        help="Base name for output files.")
    parser.add_argument("-n", "--nslices", type=int, default=1,
                        help="Number of box slices (unused in this example).")
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

    # Load Universe
    logging.info("Initializing MDAnalysis Universe...")
    u = mda.Universe(args.top, args.traj, continuous=True)
    n_frames = len(u.trajectory) if args.num is None else args.num
    group = u.select_atoms(args.Group_1)
    n_atoms = len(group)
    if n_atoms == 0:
        raise ValueError(f"No atoms matched selection '{args.Group_1}'")

    # Read positions into NumPy
    pos_list = []
    logging.info(f"Stacking {n_frames} frames of {n_atoms} atoms...")
    for i, ts in enumerate(tqdm(u.trajectory, total=n_frames)):
        if i >= n_frames: break
        pos_list.append(group.positions.copy())
    positions = np.stack(pos_list, axis=0)  # shape (n_frames, n_atoms, 3)

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

