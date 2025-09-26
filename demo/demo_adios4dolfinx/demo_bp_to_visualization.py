import flatiron_tk
import subprocess

subprocess.run(['mpirun', '-n', '2', 'python3', '_run_nse.py'])

# Convert ADIOS2 BP file to XDMF and PVD formats for visualization
flatiron_tk.bp_to_xdmf('output-bp/u.bp', 'output-xdmf/u.xdmf', name='u', time_id='all',
                    element_family='CG', element_degree=1, element_shape='vector')

flatiron_tk.bp_to_pvd('output-bp/u.bp', 'output-pvd/u.pvd', name='u', time_id='all',
                    element_family='CG', element_degree=1, element_shape='vector')
