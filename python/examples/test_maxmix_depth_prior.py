"""Behavioral verification of create_maxmix_depth_bundle_adjuster.

Run inside the dev container (needs the freshly built pycolmap):
    python3 python/examples/test_maxmix_depth_prior.py

Each check builds a tiny problem with one 3D point observed by one camera,
holds the pose and shift/scale constant, and optimizes only the point. The
max-mixture factor should pull the point's depth to the winning mode.
"""

import numpy as np
import pyceres  # must be imported before pycolmap (registers ceres::Manifold)
import pycolmap


def make_scene(depth_init):
    """Reconstruction with one identity-pose image and one point at depth_init."""
    rec = pycolmap.Reconstruction()
    camera = pycolmap.Camera(
        camera_id=1,
        model="SIMPLE_PINHOLE",
        width=100,
        height=100,
        params=[500.0, 50.0, 50.0],
    )
    rec.add_camera(camera)
    image = pycolmap.Image(image_id=1, camera_id=1, name="im0")
    image.cam_from_world = pycolmap.Rigid3d()  # identity pose
    rec.add_image(image)
    p3d_id = rec.add_point3D(
        np.array([0.0, 0.0, depth_init]), pycolmap.Track(), np.zeros(3)
    )
    return rec, p3d_id


def solver_options():
    # Tight tolerances so converged values can be compared to analytic optima;
    # ceres defaults (1e-6 function tolerance) stop a few 1e-3 short.
    options = pyceres.SolverOptions()
    options.minimizer_progress_to_stdout = False
    options.max_num_iterations = 200
    options.function_tolerance = 1e-15
    options.gradient_tolerance = 1e-15
    options.parameter_tolerance = 1e-14
    return options


def solve_maxmix(depth_init, modes, weights, sigmas):
    """Optimize the point depth under a single max-mix factor; return final z."""
    rec, p3d_id = make_scene(depth_init)
    image = rec.images[1]
    problem = pyceres.Problem()
    shift_scale = np.zeros(2)

    pycolmap.create_maxmix_depth_bundle_adjuster(
        problem,
        1,
        [p3d_id],
        np.asarray(modes, float).reshape(1, -1),
        np.asarray(weights, float).reshape(1, -1),
        np.asarray(sigmas, float).reshape(1, -1),
        [1.0],  # loss scale unused by TRIVIAL
        pycolmap.LossFunctionType.TRIVIAL,
        shift_scale,
        rec,
    )

    problem.set_parameter_block_constant(image.cam_from_world.rotation.quat)
    problem.set_parameter_block_constant(image.cam_from_world.translation)
    problem.set_parameter_block_constant(shift_scale)

    summary = pyceres.SolverSummary()
    pyceres.solve(solver_options(), problem, summary)
    return rec.points3D[p3d_id].xyz[2]


def solve_unimodal(depth_init, depth, sigma_log):
    """Same problem with the unimodal log factor (external 1/sigma^2 weighting)."""
    rec, p3d_id = make_scene(depth_init)
    image = rec.images[1]
    problem = pyceres.Problem()
    shift_scale = np.zeros(2)

    pycolmap.create_depth_bundle_adjuster(
        problem,
        1,
        [p3d_id],
        [depth],
        [1.0 / sigma_log**2],  # loss_magnitudes = inverse log-variance
        [1.0],  # loss scale unused by TRIVIAL
        pycolmap.LossFunctionType.TRIVIAL,
        shift_scale,
        rec,
        logloss=True,
    )

    problem.set_parameter_block_constant(image.cam_from_world.rotation.quat)
    problem.set_parameter_block_constant(image.cam_from_world.translation)
    problem.set_parameter_block_constant(shift_scale)

    summary = pyceres.SolverSummary()
    pyceres.solve(solver_options(), problem, summary)
    return rec.points3D[p3d_id].xyz[2]


def check(name, got, want, tol=1e-3):
    ok = abs(got - want) < tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {got:.6f}, want {want:.6f}")
    return ok


def main():
    all_ok = True

    # A) Proximity: init near the far mode -> converge to the far mode.
    z = solve_maxmix(4.5, modes=[2.0, 5.0], weights=[0.5, 0.5], sigmas=[0.1, 0.1])
    all_ok &= check("proximity selects near mode", z, 5.0)

    # B) The 2*log(sigma/w) selection term flips the winner near the boundary:
    #    same init (slightly on the far-mode side of the log-midpoint of 2 and
    #    5), same sigmas; only the weights are mirrored. Scores at init:
    #    w=[.99,.01]: 20.5 vs 21.9 -> near mode wins despite init leaning far;
    #    w=[.01,.99]: 29.7 vs 12.7 -> far mode wins.
    z = solve_maxmix(3.3, modes=[2.0, 5.0], weights=[0.99, 0.01], sigmas=[0.1, 0.1])
    all_ok &= check("high weight pulls to mode 1", z, 2.0)
    z = solve_maxmix(3.3, modes=[2.0, 5.0], weights=[0.01, 0.99], sigmas=[0.1, 0.1])
    all_ok &= check("mirrored weights pull to mode 2", z, 5.0)

    # D) K=1 reduces to the unimodal factor (same minimizer).
    z_mm = solve_maxmix(3.0, modes=[4.0], weights=[1.0], sigmas=[0.2])
    z_um = solve_unimodal(3.0, depth=4.0, sigma_log=0.2)
    all_ok &= check("K=1 matches unimodal", z_mm, z_um)
    all_ok &= check("K=1 converges to prior", z_mm, 4.0)

    # E) Degenerate duplicated mode behaves like K=1.
    z = solve_maxmix(3.0, modes=[4.0, 4.0], weights=[0.5, 0.5], sigmas=[0.2, 0.2])
    all_ok &= check("duplicated modes = unimodal", z, 4.0)

    print("\nALL PASS" if all_ok else "\nSOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
