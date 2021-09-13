# feffi module lives one level up current dir
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import feffi
from feffi.functions import energy_norm
import unittest
import csv
from fenics import norm, project, interpolate, plot

# Stored solutions have precision -5
# SIMULATION_PRECISION = -5
COMPARISON_PRECISION = 3

class FEFFIBenchmarksTestsBase(unittest.TestCase):
    def assertAlmostEqualDict(self, first, second, precision=COMPARISON_PRECISION):
        if first.keys() != second.keys():
            raise ValueError('Dicts do not have same entries')

        for key in first.keys():
            self.assertAlmostEqual(first[key], second[key], precision)

    def setup_scenario(self):
        """Cannot use unittest setUp method since we first need to set config"""

        mesh = feffi.mesh.create_mesh()
        f_spaces = feffi.functions.define_function_spaces(mesh)
        f = feffi.functions.define_functions(f_spaces)
        feffi.functions.init_functions(f)
        domain = feffi.boundaries.Domain(mesh, f_spaces)
        simul = feffi.simulation.Simulation(f, domain)
        simul.run()
        return f

    '''def compare_norms(self, f):
        """Check solutions norms wrt stored solution"""

        csv_handle = open(
            os.path.join('feffi', 'reference-solutions',
            feffi.parameters.config['plot_path'].split('/')[-1],
            'simul_data.csv'), mode='r')
        csv_reader = csv.DictReader(csv_handle)
        end_data = {key: float(val) for (key, val) in (list(csv_reader)[-1]).items()}
        end_n = int(end_data['n'])
        end_data.pop('n')
        print(end_data)

        self.assertAlmostEqualDict(end_data, feffi.functions.get_norms(f))'''

    def run_test(self, label):
        reference_path = os.path.join('feffi', 'reference-solutions', label)
        (f_old, _, _, _) = feffi.parameters.reload_status(reference_path)

        # plot_path is stored as an absolute path in previous simulation's
        # config.yml - here we mock it to become /tmp since storage of plots
        # is not needed.
        feffi.parameters.config['plot_path'] = '/tmp/feffi'

        # Run new simulation with the same config as reference one.
        f = self.setup_scenario()
        #self.compare_norms(f)

        self.assertAlmostEqual(norm(f_old['u_'], 'L2'), norm(f['u_'], 'L2'), COMPARISON_PRECISION)
        self.assertAlmostEqual(norm(f_old['p_'], 'L2'), norm(f['p_'], 'L2'), COMPARISON_PRECISION)
        self.assertAlmostEqual(norm(f_old['T_'], 'L2'), norm(f['T_'], 'L2'), COMPARISON_PRECISION)
        self.assertAlmostEqual(norm(f_old['S_'], 'L2'), norm(f['S_'], 'L2'), COMPARISON_PRECISION)

        self.assertAlmostEqual(energy_norm(f_old['u_']), energy_norm(f['u_']), COMPARISON_PRECISION)
        self.assertAlmostEqual(energy_norm(f_old['p_']), energy_norm(f['p_']), COMPARISON_PRECISION)
        self.assertAlmostEqual(energy_norm(f_old['T_']), energy_norm(f['T_']), COMPARISON_PRECISION)
        self.assertAlmostEqual(energy_norm(f_old['S_']), energy_norm(f['S_']), COMPARISON_PRECISION)

        self.assertAlmostEqual(norm(f_old['u_'].vector(), 'linf'), norm(f['u_'].vector(), 'linf'), COMPARISON_PRECISION)
        self.assertAlmostEqual(norm(f_old['p_'].vector(), 'linf'), norm(f['p_'].vector(), 'linf'), COMPARISON_PRECISION)
        self.assertAlmostEqual(norm(f_old['T_'].vector(), 'linf'), norm(f['T_'].vector(), 'linf'), COMPARISON_PRECISION)
        self.assertAlmostEqual(norm(f_old['S_'].vector(), 'linf'), norm(f['S_'].vector(), 'linf'), COMPARISON_PRECISION)


class FEFFIBenchmarksTestsQuick(FEFFIBenchmarksTestsBase):
    def test_LDC_nu_1e_2(self):
        feffi.flog.info('** Lid Driven Cavity benchmark, nu 1e-2 **')
        self.run_test('LDC_1e-2')

    def test_BDC_beta_1000(self):
        feffi.flog.info('** Buoyancy Driven Cavity benchmark, beta 1000 **')
        self.run_test('BDC_1000')

    '''# requires sill to be added in some automatic way...
    def test_Ford_steady_state(self):
        feffi.flog.info('** Ford Steady State benchmark **')
        self.run_test('FORD_STEADY_STATE')'''


def FEFFIBenchmarksTestsThorough(FEFFIBenchmarksTestsBase):
    def test_LDC_nu_5e_3(self):
        feffi.flog.info('** Lid Driven Cavity benchmark, nu 5e-3 **')
        self.run_test('LDC_5e-3')

    def test_LDC_nu_2_5e_3(self):
        feffi.flog.info('** Lid Driven Cavity benchmark, nu 2.5e-3 **')
        self.run_test('LDC_2.5e-3')

    def test_LDC_nu_1e_3(self):
        feffi.flog.info('** Lid Driven Cavity benchmark, nu 1e-3 **')
        self.run_test('LDC_1e-3')

    def test_LDC_nu_5e_4(self):
        feffi.flog.info('** Lid Driven Cavity benchmark, nu 5e-4 **')
        self.run_test('LDC_5e-4')

    def test_LDC_nu_2_5e_4(self):
        feffi.flog.info('** Lid Driven Cavity benchmark, nu 2.5e-4 **')
        self.run_test('LDC_2.5e-4')

    def test_LDC_nu_1e_4(self):
        feffi.flog.info('** Lid Driven Cavity benchmark, nu 1e-4 **')
        self.run_test('LDC_1e-4')

    def test_BDC_beta_10000(self):
        feffi.flog.info('** Buoyancy Driven Cavity benchmark, beta 10000 **')
        self.run_test('BDC_10000')

    def test_BDC_beta_10000(self):
        feffi.flog.info('** Buoyancy Driven Cavity benchmark, beta 100000 **')
        self.run_test('BDC_100000')

    def test_BDC_beta_1000000(self):
        feffi.flog.info('** Buoyancy Driven Cavity benchmark, beta 1000000 **')
        self.run_test('BDC_1000000')

    def test_RBC_beta_2500_slip(self):
        feffi.flog.info('** Rayleigh-Benard Convection benchmark, beta 2500, slip **')
        self.run_test('RBC_2500_slip')

    def test_RBC_beta_2500_noslip(self):
        feffi.flog.info('** Rayleigh-Benard Convection benchmark, beta 2500, noslip **')
        self.run_test('RBC_2500_noslip')


if __name__ == '__main__':
    unittest.main()
