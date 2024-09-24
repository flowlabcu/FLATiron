'''
Defines a scalar transport problem
'''
import fenics as fe
from abc import ABC, abstractmethod
from ..io import *

def to_fe_constant(a):
    '''
    Convert a to fe.Constant(a) if a is a number or numpy array
    '''
    if isinstance(a, float) or isinstance(a, int):
        return fe.Constant(a)
    return a


class PhysicsProblem(ABC):

    """
    This is an abstract class that defines a physics problem that we are solving.
    Any specific physics problem, e.g., transport, flow, etc. will inherit this class.

    :param mesh: The mesh domain that the physics problem is solved on
    :type mesh: flatiron_tk.Mesh
    :param tag: A tag for the physics problem name. Default is 'Var'.
    :type tag: str, optional
    """

    # -----------------------------------------
    # Init
    # -----------------------------------------
    def __init__(self, mesh, tag='u'):

        # Record inputs
        self.mesh = mesh
        self.dim = self.mesh.dim
        self.dirichlet_bcs = []

        # Get domain measure
        self.dx = fe.dx(metadata={"quadrature_degree": 4})
        self.ds = fe.ds(subdomain_data=mesh.boundary, metadata={"quadrature_degree": 4})

        # Set variable name tag
        self.tag = tag

        # Set property dictionary
        # this dictionary is for consts related to the physics state
        self.external_function_dict = {}

    def set_tag(self, tag):
        self.tag = tag

    # -----------------------------------------
    # Function space initialization
    # -----------------------------------------
    def set_element(self, element_family, element_degree):
        '''
        Set the finite element for this physics problem.

        :param element_family: A string describing the finite element type.
        :type element_family: str
        :param element_degree: An integer describing the element degree.
        :type element_degree: int
        '''
        self.element = fe.FiniteElement(element_family, self.mesh.mesh.ufl_cell(), element_degree)
        self.element_family = element_family
        self.element_degree = element_degree

    def build_function_space(self):
        """
        Build the function space and set member Test/Trial/Solution functions
        """
        self.set_function_space(fe.FunctionSpace(self.mesh.fenics_mesh(), self.element))
        self.set_test_function(fe.TestFunction(self.function_space()))
        self.set_trial_function(fe.TrialFunction(self.function_space()))
        self.set_solution_function(fe.Function(self.function_space()))

    def set_quadrature_degree(self, quad_deg):
        '''
        Set the quadrature degree for integration. Note that currently, we use the same quadrature degree for both
        the volume and surface integrals
        '''
        self.dx = fe.dx(metadata={"quadrature_degree": quad_deg})
        self.ds = fe.ds(subdomain_data=mesh.boundary, metadata={"quadrature_degree": quad_deg})

    # -----------------------------------------
    # Setters/Getters for functions
    # -----------------------------------------
    def function_space(self, physics_tag=None):
        return self.V

    def test_function(self, physics_tag=None):
        return self.test_func

    def trial_function(self, physics_tag=None):
        return self.trial_func

    def solution_function(self, physics_tag=None):
        return self.solution

    def set_function_space(self, V):
        self.V = V

    def set_test_function(self, tef):
        self.test_func = tef

    def set_trial_function(self, trf):
        self.trial_func = trf

    def set_solution_function(self, slf):
        self.solution = slf

    # -----------------------------------------
    # Weak formulation
    # -----------------------------------------
    @abstractmethod
    def set_weak_form(self):
        """
        Sets the weak formulation for the problem.

        To be implemented in derived classes.
        """
        pass

    def add_to_weakform(self, form, domain=None):
        if domain is None:
            self.weak_form += form
        else:
            self.weak_form = self.weak_form + form*domain

    def get_weak_form(self):
        return self.weak_form

    def set_external_function(self, func_name, func_val):
        self.external_function_dict[func_name] = to_fe_constant(func_val)

    def external_function(self, func_name):
        return self.external_function_dict[func_name]

    def get_zero_constant(self):
        return fe.Constant(0.)

    @abstractmethod
    def flux(self, h):
        pass

    # -----------------------------------------
    # Residue
    # -----------------------------------------
    @abstractmethod
    def get_residue(self):
        """Computes the residue of the problem.

        To be implemented in derived classes.

        Returns:
            The residue as a Fenics form.
        """
        pass

    # -----------------------------------------
    # Boundary conditions
    # -----------------------------------------
    def set_bcs(self, bcs_dict):
        '''
        Create a list of fe.DirichletBC objects based on the values supplied in bcs_dict.

        :param bcs_dict: A nested dictionary specifying boundary conditions.
                         The dictionary should have the following structure:

                         bcs_dict = {boundary_id: {``'type'``: ``'dirichlet'`` or ``'neumann'``, ``'value'``: ``value``}, ...}
        :type bcs_dict: dict
        '''
        bcs = [] # TODO: Why is this here?
        boundary = self.mesh.boundary
        for boundary_id in bcs_dict:
            bc_data = bcs_dict[boundary_id]
            bc_value = self.get_zero_constant() if bc_data['value'] == 'zero' else bc_data['value']
            if bc_data['type'] == 'dirichlet':
                self.dirichlet_bcs.append(fe.DirichletBC(self.V, bc_value, boundary, boundary_id))
            elif bc_data['type'] == 'neumann':
                flux_term = self.flux(bc_value)
                self.add_to_weakform(flux_term, self.ds(boundary_id))

    # -----------------------------------------
    # IO
    # -----------------------------------------
    def set_writer(self, directory, file_format):

        """
        Set the output file and field name for writing results.

        :param output_file: The name of the output file. The output file must ends in h5 or pvd. This will determine the format of the output
        :type output_file: str
        :param field_name: The name of the field to be written.
        :type field_name: str
        """

        # Set directory
        os.system("mkdir -p %s" % directory)

        # Set outputfile name and field name
        self.output_file = os.path.join(directory, self.tag+'.'+file_format)

        # Set output fid
        if file_format == 'h5':
            self.output_fid = self._get_h5_file(self.output_file)
        elif file_format == 'pvd':
            self.output_fid = self._get_pvd_file(self.output_file)
        else:
            raise ValueError("Incorrect output file extension")

        # Print out the path
        if self.mesh.comm.rank == 0:
            print("Output file set to %s"%self.output_file)

    def _get_h5_file(self, output_file):
        return h5_init_output_file(output_file, mesh=self.mesh.mesh, boundaries=self.mesh.boundary)

    def _get_pvd_file(self, output_file):
        return fe.File(output_file)

    def write(self, function_to_save=None, **kwargs):

        # Get time stamp if applicable
        time_stamp = kwargs.pop("time_stamp", 0.0)

        # If nothing is provided, save the solution
        if function_to_save is None:
            function_to_save = self.solution

        if self.output_file.endswith('h5'):
            h5_write(function_to_save, self.tag, h5_object=self.output_fid, timestamp=time_stamp)

        elif self.output_file.endswith('pvd'):
            function_to_save.rename(self.tag, self.tag)
            self.output_fid.write(function_to_save, time_stamp)

    def read_function_from_h5(self, h5_file, h5_group='default', time_id=None):
        if h5_group == 'default':
            h5_group = self.tag
        V = self.function_space()
        return h5_mod.h5_read(h5_file, h5_group, 'function', mesh=self.mesh.fenics_mesh(), function_space=V, time_id=time_id)








