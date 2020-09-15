#include <dolfin.h>
#include "NonlinearNS.h"
#include "Functionals.h"

using namespace dolfin;

struct MeshOut
{
    std::shared_ptr<dolfin::Mesh> mesh;
    std::shared_ptr<dolfin::MeshFunction<std::size_t>> boundaries;
};

// inlet
class InflowVelocity : public Expression
{
public:
    // Constructor
    InflowVelocity() : Expression(3) {}

    // Evaluate velocity at inflow
    void eval(Array<double> &values, const Array<double> &x) const
    {
        values[0] = 5.0 * (1 - 4 * x[2] * x[2] - 4 * x[1] * x[1]);
        values[1] = 0;
        values[2] = 0;
    }
};

MeshOut get_mesh()
{
    auto mesh = std::make_shared<Mesh>();
    XDMFFile initialmeshfile(MPI_COMM_WORLD, "../../geometry/mesh/designB-reverse/domain.xdmf");
    initialmeshfile.read(*mesh);

    MeshValueCollection<std::size_t> mf_mvc(mesh, 2);
    XDMFFile initialmffile(MPI_COMM_WORLD, "../../geometry/mesh/designB-reverse/boundaries.xdmf");
    initialmffile.read(mf_mvc, "boundaries");
    auto boundaries = std::make_shared<MeshFunction<std::size_t>>(mesh, mf_mvc);

    return {.mesh = mesh, .boundaries = boundaries};
};

double inlet_pressure(MeshOut meshout, std::shared_ptr<dolfin::Function> w)
{
    auto mesh = meshout.mesh;
    auto boundaries = meshout.boundaries;

    Functionals::Form_inlet_pressure pressure(mesh, w);
    pressure.ds = boundaries;

    return assemble(pressure) / (3.14 * 0.5 * 0.5);
}

int main()
{
    // Print log messages only from the root process in parallel
    parameters["std_out_all_processes"] = false;

    // Set parameter values
    double dt = 0.05;
    double T = 10;
    double mu = 0.001;
    double rho = 1;
    // Crank-Nicholson
    double theta = 0.5;

    // Load mesh from file
    auto meshout = get_mesh();
    auto mesh = meshout.mesh;
    auto boundaries = meshout.boundaries;

    auto W = std::make_shared<NonlinearNS::FunctionSpace>(mesh);

    // Create boundary conditions
    auto v_in = std::make_shared<InflowVelocity>();
    auto p_out = std::make_shared<Constant>(0.0);
    auto zero_vec = std::make_shared<Constant>(0.0, 0.0, 0.0);

    auto inflow = std::make_shared<const DirichletBC>(W->sub(0), v_in, boundaries, 2);
    auto outflow = std::make_shared<const DirichletBC>(W->sub(1), p_out, boundaries, 3);

    auto noslip = std::make_shared<const DirichletBC>(W->sub(0), zero_vec, boundaries, 4);

    auto bcs = {noslip, inflow, outflow};

    // Create forms
    auto F = std::make_shared<NonlinearNS::ResidualForm>(W);
    auto J = std::make_shared<NonlinearNS::JacobianForm>(W, W);

    // Create functions
    auto w0 = std::make_shared<Function>(W);
    auto w = std::make_shared<Function>(W);

    // Create coefficients
    auto k = std::make_shared<Constant>(dt);
    auto f = std::make_shared<Constant>(0, 0, 0);
    auto nu = std::make_shared<Constant>(mu / rho);
    auto theta_c = std::make_shared<Constant>(theta);

    // Set coefficients
    std::map<std::string, std::shared_ptr<const GenericFunction>> coefficients =
        {{"w", w}, {"w0", w0}, {"k", k}, {"nu", nu}, {"theta", theta_c}, {"f", f}};
    J->set_coefficients(coefficients);
    F->set_coefficients(coefficients);

    File vfile("velocity.pvd");
    File pfile("pressure.pvd");

    // Time-stepping
    double t = 0;
    int count = 0;
    int timesteps_saved = 120;
    int save_every = ((T / dt) / timesteps_saved) == 0 ? 1 : (T / dt) / timesteps_saved;
    auto problem = std::make_shared<NonlinearVariationalProblem>(F, w, bcs, J);
    auto solver = NonlinearVariationalSolver(problem);

    // snes solver -- comment out to use newton solver
    solver.parameters["nonlinear_solver"] = "snes";
    solver.parameters("snes_solver")["line_search"] = "bt";
    solver.parameters("snes_solver")["linear_solver"] = "mumps";
    solver.parameters("snes_solver")["preconditioner"] = "hypre_amg";
    solver.parameters("snes_solver")["maximum_iterations"] = 500;
    solver.parameters("snes_solver")["absolute_tolerance"] = 1E-5;
    solver.parameters("snes_solver")["relative_tolerance"] = 1E-5;
    solver.parameters("snes_solver")["report"] = true;

    char time_stdout_buffer[50];
    double inlet_p;
    while (t < T + DOLFIN_EPS)
    {
        sprintf(time_stdout_buffer, "CURRENT TIME: %f", t);
        info(time_stdout_buffer);

        *w0->vector() = *w->vector();

        solver.solve();

        // Save to file
        if (count % save_every == 0)
        {
            vfile << (*w0)[0];
            pfile << (*w0)[1];
        }

        // Move to next time step
        t += dt;
        count++;

        inlet_p = inlet_pressure(meshout, w);

        sprintf(time_stdout_buffer, "AVERAGE INLET PRESSURE: %f", inlet_p);
        info(time_stdout_buffer);
    }

    return 0;
}
