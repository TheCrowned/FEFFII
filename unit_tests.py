import os
if not os.path.isdir('feffi'): #if not already in right place - hacky!
    os.chdir("../") #for feffi to work
import feffi
import unittest
import csv
from fenics import norm



class FEFFIBenchmarksTests(unittest.TestCase):
    def assertAlmostEqualDict(self, first, second, precision=3):
        if first.keys() != second.keys():
            raise ValueError('Dicts do not have same entries')

        for key in first.keys():
            self.assertAlmostEqual(first[key], second[key], precision)

    def setupScenario(self):
        mesh = feffi.mesh.create_mesh()
        f_spaces = feffi.functions.define_function_spaces(mesh)
        f = feffi.functions.define_functions(f_spaces)
        feffi.functions.init_functions(f)
        domain = feffi.boundaries.Domain(mesh, f_spaces)
        simul = feffi.simulation.Simulation(f, domain.BCs)
        simul.run()
        return f

    def compareNorms(self, f):
        # Check norms wrt stored solution
        csv_handle = open(os.path.join('feffi', 'reference-solutions', feffi.parameters.config['plot_path'].split('/')[-1], 'simul_data.csv'), mode='r')
        csv_reader = csv.DictReader(csv_handle)
        end_data = {key: float(val) for (key, val) in (list(csv_reader)[-1]).items()}
        end_n = int(end_data['n'])
        end_data.pop('n')

        self.assertAlmostEqualDict(end_data, feffi.functions.get_norms(f))

    def test_LDC_nu_1e_2(self):
        feffi.flog.info('** Lid Driven Cavity benchmark, nu 1e-2 **')
        feffi.parameters.define_parameters({
            'config_file': 'feffi/config/lid-driven-cavity.yml',
            'nu': [1e-2],
            'plot_path': 'LDC_1e-2',
        })

        f = self.setupScenario()
        self.compareNorms(f)

    def test_BDC_beta_1000(self):
        feffi.flog.info('** Buoyancy Driven Cavity benchmark, beta 1000 **')
        feffi.parameters.define_parameters({
            'config_file': 'feffi/config/buoyancy-driven-cavity.yml',
            'beta': 1000,
            'plot_path': 'BDC_1000',
        })

        f = self.setupScenario()
        self.compareNorms(f)


if __name__ == '__main__':
    unittest.main()
