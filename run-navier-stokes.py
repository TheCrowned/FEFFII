from navierstokes import NavierStokes

ns = NavierStokes()

ns.create_mesh()
ns.define_function_spaces()
ns.define_variational_problems()
ns.boundary_conditions()

if ns.args.domain == 'custom':
	ns.mesh_add_sill(ns.args.domain_size_x/2, ns.args.domain_size_y/5, ns.args.domain_size_x/5)

ns.run_simulation()
