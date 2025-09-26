import adios4dolfinx
import basix
import dolfinx
import numbers
import subprocess
import ufl 

from abc import ABC, abstractmethod
from collections.abc import Iterable
from flatiron_tk.fem import boundary_conditions as bcs
from flatiron_tk.io import *
from pathlib import Path

def _is_number(a):
    """
    Check if 'a' is a number.
    Parameters
    ----------
    a : any
        The input to check.   
    Returns
    -------
    bool
        True if 'a' is a number, False otherwise."""
    return isinstance(a, numbers.Number)

def _is_iterable_number(a):
    """
    Check if 'a' is an iterable of numbers.
    Parameters
    ----------
    a : iterable
        The input to check.   
    Returns
    -------
    bool
        True if 'a' is an iterable of numbers, False otherwise.
    """
    # Return false if not iterable
    if not isinstance(a, Iterable):
        return False
    
    # Check for strings (considered iterable, so we create a safeguard)
    if isinstance(a, (str, bytes)):
        return False

    # Check for numbers
    return all(isinstance(ai, numbers.Number) for ai in a) 

class PhysicsProblem(ABC):
    """
    Abstract base class for defining physics problems using the finite element method.
    This class provides a framework for setting up and solving physics problems on a given mesh.
    It includes methods for defining the finite element, function space, weak form, boundary conditions,
    and writing results to output files.
    Inherited classes must implement the `set_weak_form` method to define the specific weak form of the problem.

    Parameters
    ----------
    mesh : flatiron_tk mesh object
        The mesh object representing the computational domain.
    tag : str, optional
        Tag for the solution function (default is 'u').
    q_degree : int, optional
        Quadrature degree for the function space (default is 4).
    **kwargs
        Additional keyword arguments.

    Attributes
    ----------
    dirichlet_bcs : list
        List to store Dirichlet boundary conditions.
    mesh : flatiron_tk mesh object
        The mesh object representing the computational domain.
    number_of_steps_written : int
        Counter for the number of steps written.
    tag : str
        Tag for the solution function.
    q_degree : int
        Quadrature degree for the function space.
    dx : ufl.Measure
        Volume differential (domain measure).
    ds : ufl.Measure
        Exterior surface differential.
    dS : ufl.Measure
        Interior surface differential.
    external_function_dict : dict
        Dictionary to hold constant physical properties as external functions.
    
    """

    def __init__(self, mesh, tag='u', q_degree=4, **kwargs):
        self.dirichlet_bcs = []
        self.mesh = mesh 
        self.number_of_steps_written = 0 
        self.tag = tag
        self.q_degree = q_degree        

        # ---- Domain measures ---- #
        # dx is the volume differential (or domain differential), dOmega
        self.dx = ufl.Measure('dx', metadata={'quadrature_degree': self.q_degree})
        # ds is the EXTERIOR surface differential, dGamma
        self.ds = ufl.Measure('ds', domain=self.mesh.msh, subdomain_data=self.mesh.boundary, metadata={'quadrature_degree': self.q_degree})
        # dS is the INTERIOR surface differential, dGamma0
        self.dS = ufl.Measure('dS', domain=self.mesh.msh, metadata={'quadrature_degree': self.q_degree})
        # The constant physical properties will be held in the external function dictionary 
        self.external_function_dict = {}

    def set_tag(self, tag):
        """
        Parameters
        ----------
        tag: The tag to assign to the solution function.
        
        """
        
        print(f'Setting tag: {tag}')
        self.tag = tag

    def to_dolfinx(self, a):
        """
        Convert a to a dolfinx constant if a is a number or an iterable, otherwise return a.

        Parameters
        ----------
        a : number or iterable or ufl expression or dolfinx function/constant
            The input to convert.   
        
        Returns
        -------
        dolfinx.fem.Constant or ufl expression or dolfinx function/constant
            The converted dolfinx constant or the original input if it was already a ufl expression or dolfinx function/constant.
        
        Raises
        ------
        ValueError
            If the input is not a number, iterable, ufl expression, or dolfinx function/constant.   
        
        """
        # If 'a' is a ufl expression, return it as is
        if isinstance(a, ufl.core.expr.Expr):
            return a
    
        # If 'a' is a dolfinx function or constant, return it as is
        if isinstance(a, (dolfinx.fem.Function, dolfinx.fem.Constant)):
            return a
        
        # If 'a' is a number or an iterable of numbers, convert it to a dolfinx constant
        if _is_number(a) or _is_iterable_number(a):
            return dolfinx.fem.Constant(self.mesh.msh, dolfinx.default_scalar_type(a))
        return a
        
    def set_element(self, element_family, element_degree, element_shape=None):
        """
        Set the finite element for the physics problem.
        
        Parameters
        ----------
        element_family : str
            The family of the finite element (e.g., "CG", "DG").
        element_degree : int
            The degree of the finite element.
        element_shape : str, optional
            The shape of the finite element (e.g., "scalar", "vector"). Default is None.
        Raises
        ------
        """

        # Set the dolfinx shape for vectors and tensors
        _shape = element_shape
        if element_shape == 'vector':
            _shape = (self.mesh.get_tdim(), )
        elif element_shape == 'tensor':
            _shape = (self.mesh.get_tdim(), self.mesh.get_tdim())

        # Set the dolfinx element
        self.element = basix.ufl.element(element_family, self.mesh.msh.basix_cell(), element_degree, shape=_shape)
        self.element_family = element_family
        self.element_degree = element_degree
        self.element_shape = _shape

    def set_quadrature_degree(self, q_degree):
        """
        Set the quadrature degree for the physics problem.
        
        Parameters
        ----------
        q_degree : int
            The quadrature degree to set.
        """
        self.q_degree = q_degree
        self.dx = ufl.Measure('dx', metadata={'quadrature_degree': self.q_degree})
        self.ds = ufl.Measure('ds', domain=self.mesh.msh, subdomain_data=self.mesh.boundary, metadata={'quadrature_degree': self.q_degree})
        self.dS = ufl.Measure('dS', domain=self.mesh.msh, metadata={'quadrature_degree': self.q_degree})
    
    def build_function_space(self):
        """
        Build the function space, test function, trial function, and solution function for the physics problem.
        """
        self.set_function_space(dolfinx.fem.functionspace(self.mesh.msh, self.element)) 
        self.set_test_function(ufl.TestFunction(self.get_function_space()))
        self.set_trial_function(ufl.TrialFunction(self.get_function_space()))
        self.set_solution_function(dolfinx.fem.Function(self.get_function_space()))

    # ---- Setters and Getters ---- #
    def set_external_function(self, function_name, function):
        """
        Set an external function for the physics problem. 
        
        Parameters
        ----------
        function_name : str
            The name of the external function.
        function : number or iterable or ufl expression or dolfinx function/constant
            The external function to set. Can be a number, an iterable of numbers, a ufl expression, or a dolfinx function/constant.
        
        """
        self.external_function_dict[function_name] = self.to_dolfinx(function)
    
    def set_function_space(self, V):
        """
        Set the function space for the physics problem.
        Parameters
        ----------
        V : dolfinx.fem.FunctionSpace
            The function space to set.
        """
        self.V = V

    def set_test_function(self, test_function):
        """
        Set the test function for the physics problem.
        Parameters
        ----------
        test_function : ufl.TestFunction
            The test function to set.
        """
        self.test_function = test_function

    def set_trial_function(self, trial_function):
        """
        Set the trial function for the physics problem.
        Parameters
        ----------
        trial_function : ufl.TrialFunction
            The trial function to set.
        """
        self.trial_function = trial_function

    def set_solution_function(self, solution_function):
        """
        Set the solution function for the physics problem.
        Parameters
        ----------
        solution_function : dolfinx.fem.Function
            The solution function to set.
        """
        self.solution = solution_function

    @abstractmethod
    def set_weak_form(self):
        """
        Set the weak form of the physics problem.

        This is done in the inherited class.
        """
        pass

    def flux(self, h):
        """
        Define the flux term for Neumann boundary conditions.
        Parameters
        ----------
        h : number or iterable or ufl expression or dolfinx function/constant
            The flux value. Can be a number, an iterable of numbers, a ufl expression, or a dolfinx function/constant.
        """
        pass

    def external_function(self, function_name):
        """
        Get an external function by name.
        Parameters
        ----------
        function_name : str
            The name of the external function to get.   
        
        Returns
        -------
        dolfinx.fem.Constant or ufl expression or dolfinx function/constant
            The external function associated with the given name.
        """
        return self.external_function_dict[function_name]
    
    def get_function_space(self, physics_tag=None):
        """
        Get the function space for the physics problem.
        Parameters
        ----------
        physics_tag : str, optional
            The tag of the physics problem (default is None).
        Returns
        -------
        dolfinx.fem.FunctionSpace
            The function space of the physics problem.
        """
        return self.V
    
    def get_test_function(self, physics_tag=None):
        """
        Get the test function for the physics problem.
        Parameters
        ----------
        physics_tag : str, optional
            The tag of the physics problem (default is None).
        Returns
        -------
        ufl.TestFunction
            The test function of the physics problem.
        """
        return self.test_function
    
    def get_trial_function(self, physics_tag=None):
        """
        Get the trial function for the physics problem.
        Parameters
        ----------
        physics_tag : str, optional
            The tag of the physics problem (default is None).
        Returns
        -------
        ufl.TrialFunction
            The trial function of the physics problem.
        """
        return self.trial_function
    
    def get_solution_function(self, physics_tag=None):
        """
        Get the solution function for the physics problem.
        Parameters
        ----------
        physics_tag : str, optional
            The tag of the physics problem (default is None).       
        Returns
        -------
        dolfinx.fem.Function
            The solution function of the physics problem.
        """
        return self.solution
    
    def get_weak_form(self):
        """
        Get the weak form of the physics problem.
        Returns
        -------
        ufl.Form
            The weak form of the physics problem.
        """
        return self.weak_form
    
    def jacobian(self):
        """
        Get the Jacobian of the weak form.
        Returns
        -------
        ufl.Form
            The Jacobian of the weak form.
        """
        return ufl.derivative(self.get_weak_form(), self.get_solution_function(), self.get_trial_function())
    
    def add_to_weak_form(self, form, domain=None):
        """
        Add a term to the weak form of the physics problem.
        
        Parameters
        ----------
        form : ufl.Form
            The term to add to the weak form.
        domain : ufl.Measure, optional
            The domain to which the term applies (default is None, meaning the user has specified the domain in the form.
        """
        if domain is None:
            self.weak_form += form
        else:
            self.weak_form += form * domain

    def get_residual(self):
        """
        Get the residual of the weak form.
        
        Returns
        ---------
        ufl.Form
            The residual of the weak form.
        
        In this base class, this method is not implemented and should be overridden in derived classes.
        """
        pass
        
    def set_bcs(self, bcs_dict):
        """
        Set the boundary conditions for the physics problem.

        Parameters
        ----------
        bcs_dict : dict
            A dictionary containing the boundary conditions. The keys are the boundary IDs, and the values are dictionaries with 'type' and 'value' keys.

        Raises
        ------
        ValueError
            If an unsupported boundary condition type is provided.

        """
        for boundary_id in bcs_dict:
            bc_data = bcs_dict[boundary_id]            
            bc_value = dolfinx.fem.Constant(self.mesh.msh, dolfinx.default_scalar_type(0.0)) if bc_data['value'] == 'zero' else bc_data['value']

            if bc_data['type'] == 'dirichlet':
                bc = bcs.build_dirichlet_bc(self.mesh, boundary_id, bc_value, self.get_function_space())
                self.dirichlet_bcs.append(bc)
 
            elif bc_data['type'] == 'neumann':
                flux_term = self.flux(bc_value)
                self.add_to_weak_form(flux_term, self.ds(boundary_id))
            
            # Robin BCs are added to the weak form (same as Neumann). 
            # The user must define the flux method accordingly. 
            elif bc_data['type'] == 'robin':
                flux_term = self.flux(bc_value)
                self.add_to_weak_form(flux_term, self.ds(boundary_id))
            
            else: 
                raise ValueError(f"Unsupported boundary condition type: {bc_data['type']}. Supported types are 'dirichlet' and 'neumann'.")
            
    def set_writer(self, output_dir, file_format):
        """
        Set up the file writer for the physics problem.
        
        Parameters
        ----------
        output_dir : str
            The directory where the output files will be saved.
        file_format : str
            The file format for the output files. Supported formats are 'xdmf', 'pvd', and 'bp'.
        
        Raises
        ------
        ValueError
            If an unsupported file format is provided.
        """
        subprocess.run(['mkdir', '-p', output_dir])
        self.output_file = Path(output_dir) / f'{self.tag}.{file_format}'

        if self.output_file.suffix == '.xdmf':
            self.output_fid = dolfinx.io.XDMFFile(self.mesh.msh.comm, self.output_file, 'w', encoding=dolfinx.io.XDMFFile.Encoding.HDF5)
            self.output_fid.write_mesh(self.mesh.msh) 
        
        elif self.output_file.suffix == '.pvd':
            self.output_fid = dolfinx.io.VTKFile(self.mesh.msh.comm, self.output_file, 'w')
        
        elif self.output_file.suffix == '.bp':
            self.output_fid = None # ADIOS2 handles the function name in write method 
            adios4dolfinx.write_mesh(self.output_file, self.mesh.msh, mode=adios4dolfinx.adios2_helpers.adios2.Mode.Write)

        else:
            raise ValueError(f'Unsupported file format: {file_format}. Supported formats are "xdmf" and "pvd".')
        
        if self.mesh.msh.comm.rank == 0:
            print(f'Output file set to: {self.output_file}')

    def write(self, function_to_write=None, **kwargs):
        """
        Write the solution function to the output file.

        Parameters
        ----------
        function_to_write : dolfinx.fem.Function, optional
            The function to write to the output file. If None, the solution function is written (default is None).
        
        **kwargs : dict
            Additional keyword arguments. Supported keys:

            time_stamp : float, optional
                The time stamp to associate with the output (default is the number of steps written)

        """

        time_stamp = kwargs.get('time_stamp', self.number_of_steps_written)
        self.number_of_steps_written += 1

        if function_to_write is None:
            function_to_write = self.get_solution_function()
        
        function_to_write.x.scatter_forward()
        
        # XDMF expects time — default to 0.0 if steady
        if self.output_file.suffix == '.xdmf':
            if time_stamp is None:
                time_stamp = 0.0
            self.output_fid.write_function(function_to_write, t=time_stamp)

        elif self.output_file.suffix == '.pvd':                    
            if time_stamp is not None:
                print(f'Writing function {function_to_write.name} at time {time_stamp}')
                self.output_fid.write_function(function_to_write, t=time_stamp)
            else:
                self.output_fid.write_function(function_to_write)

            self.output_fid.close()
        
        elif self.output_file.suffix == '.bp':
            if time_stamp is None:
                time_stamp = 0.0
            
            adios4dolfinx.write_function(self.output_file, function_to_write, time=time_stamp, name=function_to_write.name, 
                                                       mode=adios4dolfinx.adios2_helpers.adios2.Mode.Append) 
          