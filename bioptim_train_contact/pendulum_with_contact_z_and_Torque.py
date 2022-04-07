"""
A very simple yet meaningful optimal control program consisting in a pendulum starting downward and ending upward
while requiring the minimum of generalized forces. The solver is only allowed to move the pendulum sideways.

This simple example is a good place to start investigating bioptim as it describes the most common dynamics out there
(the joint torque driven), it defines an objective function and some boundaries and initial guesses

During the optimization process, the graphs are updated real-time (even though it is a bit too fast and short to really
appreciate it). Finally, once it finished optimizing, it animates the model using the optimal solution
"""

import biorbd_casadi as biorbd
from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    Bounds,
    BoundsList,
    QAndQDotBounds,
    InitialGuess,
    InitialGuessList,
    ConstraintFcn,
    ConstraintList,
    ObjectiveFcn,
    Objective,
    Node,
    OdeSolver,
    CostType,
    Solver,
)


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolver = OdeSolver.RK4(),
    use_sx: bool = True,
    n_threads: int = 1,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    final_time: float
        The time in second required to perform the task
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program
    ode_solver: OdeSolver = OdeSolver.RK4()
        Which type of OdeSolver to use
    use_sx: bool
        If the SX variable should be used instead of MX (can be extensive on RAM)
    n_threads: int
        The number of threads to use in the paralleling (1 = no parallel computing)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="tau")  # we tried to minimize the state
    # instead of the control

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DERIVATIVE_DRIVEN, with_contact=True)

    # Torque
    tau_min, tau_max, tau_init = -300, 300, 0
    n_tau = biorbd_model.nbGeneralizedTorque()
    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))
    x_bounds[0].concatenate(
        Bounds([tau_min] * biorbd_model.nbGeneralizedTorque(), [tau_max] * biorbd_model.nbGeneralizedTorque())
    )  # we have added the torque to the state
    x_bounds[0][:, [0, -1]] = 0
    x_bounds[0][2, -1] = 3.14
    x_bounds[0].min[3:6, 1] = -600
    x_bounds[0].max[3:6, 1] = 600
    x_bounds[0][[6, 7], :] = 0  # Prevent the model from actively rotate

    # Initial guess
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    x_init = InitialGuess([0] * (n_q + n_qdot + n_tau))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min * 100] * n_tau, [tau_max * 100] * n_tau)
    u_bounds[0][[1, 2], :] = 0  # Prevent the model from actively transpose along z and rotate long x
    u_init = InitialGuessList()
    u_init.add([tau_init * 10] * n_tau)

    constraints = ConstraintList()
    constraints.add(ConstraintFcn.TRACK_MARKERS_VELOCITY, marker_index="marker_1", axes=2, target=0, node=Node.START)
    constraints.add(ConstraintFcn.TRACK_MARKERS, marker_index="marker_1", axes=2, target=0, node=Node.START)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
        use_sx=use_sx,
        n_threads=n_threads,
    )


def main():
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(biorbd_model_path="models/pendulum_with_contact_z.bioMod", final_time=1, n_shooting=30)

    # Custom plots
    ocp.add_plot_penalty(CostType.ALL)

    # --- Print ocp structure --- #
    ocp.print(to_console=False, to_graph=False)

    # --- Solve the ocp --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))
    # sol.graphs()

    # --- Show the results in a bioviz animation --- #
    sol.print()
    sol.graphs(show_bounds=True)
    sol.animate(n_frames=100)


if __name__ == "__main__":
    main()
