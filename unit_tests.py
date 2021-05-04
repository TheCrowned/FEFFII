import os
if not os.path.isdir('feffi'): #if not already in right place - hacky!
    os.chdir("../") #for feffi to work
import feffi
import unittest
import csv
from fenics import norm

# Stored solutions have precision -5
TEST_PRECISION = -1

class FEFFIBenchmarksTests(unittest.TestCase):
    def assertAlmostEqualDict(self, first, second, precision=3):
        if first.keys() != second.keys():
            raise ValueError('Dicts do not have same entries')

        for key in first.keys():
            self.assertAlmostEqual(first[key], second[key], precision)

    def setup_scenario(self):
        mesh = feffi.mesh.create_mesh()
        f_spaces = feffi.functions.define_function_spaces(mesh)
        f = feffi.functions.define_functions(f_spaces)
        feffi.functions.init_functions(f)
        domain = feffi.boundaries.Domain(mesh, f_spaces)
        simul = feffi.simulation.Simulation(f, domain.BCs)
        simul.run()
        return f

    def compare_norms(self, f):
        """Check solutions norms wrt stored solution"""

        csv_handle = open(
            os.path.join('feffi', 'reference-solutions',
            feffi.parameters.config['plot_path'].split('/')[-1],
            'simul_data.csv'), mode='r')
        csv_reader = csv.DictReader(csv_handle)
        end_data = {key: float(val) for (key, val) in (list(csv_reader)[-1]).items()}
        end_n = int(end_data['n'])
        end_data.pop('n')

        self.assertAlmostEqualDict(end_data, feffi.functions.get_norms(f))

    def run_test(self):
        f = self.setup_scenario()
        #self.compare_norms(f)

        reference_path = os.path.join(
            'feffi', 'reference-solutions',
            feffi.parameters.config['plot_path'].split('/')[-1])
        (f_old, _, _, _) = feffi.parameters.reload_status(reference_path)

        self.assertAlmostEqual(norm(f_old['u_'], 'L2'), norm(f['u_'], 'L2'))
        self.assertAlmostEqual(norm(f_old['p_'], 'L2'), norm(f['p_'], 'L2'))
        self.assertAlmostEqual(norm(f_old['T_'], 'L2'), norm(f['T_'], 'L2'))
        self.assertAlmostEqual(norm(f_old['S_'], 'L2'), norm(f['S_'], 'L2'))

    def test_LDC_nu_1e_2(self):
        feffi.flog.info('** Lid Driven Cavity benchmark, nu 1e-2 **')
        feffi.parameters.define_parameters({
            'config_file': 'feffi/config/lid-driven-cavity.yml',
            'simulation_precision': TEST_PRECISION,
            'nu': [1e-2],
            'plot_path': 'LDC_1e-2',
        })
        self.run_test()

    def test_LDC_nu_5e_3(self):
        feffi.flog.info('** Lid Driven Cavity benchmark, nu 5e-3 **')
        feffi.parameters.define_parameters({
            'config_file': 'feffi/config/lid-driven-cavity.yml',
            'simulation_precision': TEST_PRECISION,
            'nu': [5e-3],
            'plot_path': 'LDC_5e-3',
        })
        self.run_test()

    def test_LDC_nu_2_5e_3(self):
        feffi.flog.info('** Lid Driven Cavity benchmark, nu 2.5e-3 **')
        feffi.parameters.define_parameters({
            'config_file': 'feffi/config/lid-driven-cavity.yml',
            'simulation_precision': TEST_PRECISION,
            'nu': [2.5e-3],
            'plot_path': 'LDC_2.5e-3',
        })
        self.run_test()

    def test_LDC_nu_1e_3(self):
        feffi.flog.info('** Lid Driven Cavity benchmark, nu 1e-3 **')
        feffi.parameters.define_parameters({
            'config_file': 'feffi/config/lid-driven-cavity.yml',
            'simulation_precision': TEST_PRECISION,
            'nu': [1e-3],
            'plot_path': 'LDC_1e-3',
        })
        self.run_test()

    def test_LDC_nu_5e_4(self):
        feffi.flog.info('** Lid Driven Cavity benchmark, nu 5e-4 **')
        feffi.parameters.define_parameters({
            'config_file': 'feffi/config/lid-driven-cavity.yml',
            'simulation_precision': TEST_PRECISION,
            'nu': [5e-4],
            'plot_path': 'LDC_5e-4',
        })
        self.run_test()

    def test_LDC_nu_2_5e_4(self):
        feffi.flog.info('** Lid Driven Cavity benchmark, nu 2.5e-4 **')
        feffi.parameters.define_parameters({
            'config_file': 'feffi/config/lid-driven-cavity.yml',
            'simulation_precision': TEST_PRECISION,
            'nu': [2.5e-4],
            'plot_path': 'LDC_2.5e-4',
        })
        self.run_test()

    def test_LDC_nu_1e_4(self):
        feffi.flog.info('** Lid Driven Cavity benchmark, nu 1e-4 **')
        feffi.parameters.define_parameters({
            'config_file': 'feffi/config/lid-driven-cavity.yml',
            'simulation_precision': TEST_PRECISION,
            'nu': [1e-4],
            'plot_path': 'LDC_1e-4',
        })
        self.run_test()

    def test_BDC_beta_1000(self):
        feffi.flog.info('** Buoyancy Driven Cavity benchmark, beta 1000 **')
        feffi.parameters.define_parameters({
            'config_file': 'feffi/config/buoyancy-driven-cavity.yml',
            'simulation_precision': TEST_PRECISION,
            'beta': 1000,
            'plot_path': 'BDC_1000',
        })
        self.run_test()

    def test_BDC_beta_10000(self):
        feffi.flog.info('** Buoyancy Driven Cavity benchmark, beta 10000 **')
        feffi.parameters.define_parameters({
            'config_file': 'feffi/config/buoyancy-driven-cavity.yml',
            'simulation_precision': TEST_PRECISION,
            'beta': 10000,
            'plot_path': 'BDC_10000',
        })
        self.run_test()

    def test_BDC_beta_10000(self):
        feffi.flog.info('** Buoyancy Driven Cavity benchmark, beta 100000 **')
        feffi.parameters.define_parameters({
            'config_file': 'feffi/config/buoyancy-driven-cavity.yml',
            'simulation_precision': TEST_PRECISION,
            'beta': 100000,
            'plot_path': 'BDC_100000',
        })
        self.run_test()

    def test_BDC_beta_1000000(self):
        feffi.flog.info('** Buoyancy Driven Cavity benchmark, beta 1000000 **')
        feffi.parameters.define_parameters({
            'config_file': 'feffi/config/buoyancy-driven-cavity.yml',
            'simulation_precision': TEST_PRECISION,
            'steps_n': 500,
            'beta': 1000000,
            'plot_path': 'BDC_1000000',
        })
        self.run_test()

    def test_RBC_beta_2500_slip(self):
        feffi.flog.info('** Rayleigh-Benard Convection benchmark, beta 2500, slip **')
        feffi.parameters.define_parameters({
            'config_file': 'feffi/config/rayleigh-benard-convection-slip.yml',
            'simulation_precision': TEST_PRECISION,
            #'steps_n': 500,
            'beta': 2500,
            'plot_path': 'RBC_2500_slip',
        })
        self.run_test()

    def test_RBC_beta_2500_noslip(self):
        feffi.flog.info('** Rayleigh-Benard Convection benchmark, beta 2500, noslip **')
        feffi.parameters.define_parameters({
            'config_file': 'feffi/config/rayleigh-benard-convection-noslip.yml',
            'simulation_precision': TEST_PRECISION,
            #'steps_n': 500,
            'beta': 2500,
            'plot_path': 'RBC_2500_noslip',
        })
        self.run_test()

    def test_Ford_experiment(self):
        feffi.flog.info('** Ford Steady State benchmark **')
        feffi.parameters.define_parameters({
            'config_file': 'feffi/config/ford-experiment.yml',
            'simulation_precision': TEST_PRECISION,
            'plot_path': 'FORD_STEADY_STATE',
        })
        self.run_test()

if __name__ == '__main__':
    unittest.main()
