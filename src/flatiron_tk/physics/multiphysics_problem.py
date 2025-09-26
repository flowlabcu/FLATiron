import basix
import dolfinx
import numpy as np
import os
import subprocess
import ufl

from collections.abc import Iterable
from flatiron_tk.io import *
from flatiron_tk.physics import PhysicsProblem
from mpi4py import MPI


def _is_container(obj):
    """
    Check if the object is a container (like list, tuple, set, etc.) but not a string or bytes.
    Parameters
    ----------
    obj : any
        The object to check.
    Returns
    -------
    bool
        True if the object is a container, False otherwise.
    """
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))

class MultiphysicsProblem(PhysicsProblem):
    """
    A class to represent a multiphysics problem by combining multiple physics problems.

    Initialize a multiphysics problem by combining multiple physics problems.
        Parameters
        ----------
        \*physics_problems : `PhysicsProblem`
            A variable number of physics problem instances to be combined into a multiphysics problem.
        
        Raises
        ------
        ValueError
            If the provided physics problems do not share the same mesh or if there are tag conflicts.
    
    """

    def __init__(self, *physics_problems):
        # Check if all physics problems are defined on the same mesh
        mesh_hash = hash(physics_problems[0].mesh.msh)
        for physics in physics_problems:
            if hash(physics.mesh.msh) != mesh_hash:
                raise ValueError('All physics problems must share the same mesh.')
            
        # Put the subphysics problems in a list
        self.sub_physics = list(physics_problems)
        self.num_sub_physics = len(self.sub_physics)

        # Initialize the joint monolithic multiphysics problem using the mesh 
        # of the first subphysics problem (they're all the same mesh)
        super().__init__(physics_problems[0].mesh)

        # Set a unique tag for each subphysics problem/solution 
        self.tag = {}
        self._is_mixed_element = []
        physics_id = 0

        for physics in self.sub_physics:
            # Check to see if the physics problem already has a dictionary 
            # and therefore already is a multiphysics problem
            if isinstance(physics.tag, dict):
                self._is_mixed_element.append(True)

                # Check to see if the sub_physics uses an already existing tag
                for tag in physics.tag:
                    if tag in self.tag.keys():
                        raise ValueError(f'Tag {tag} already exists in the multiphysics problem. \
                                         Please set a unique tag for each type of physics.')
                    # Create a dicitionary entry for the tag with the key as the tag and 
                    # the value as the physics_id
                    self.tag[tag] = physics_id 
                
                physics_id += 1
            
            else:
                self._is_mixed_element.append(False)
                # Check to see if the sub_physics uses an already existing tag
                if physics.tag in self.tag.keys():
                    raise ValueError(f'Tag {physics.tag} already exists in the multiphysics problem. \
                                     Please set a unique tag for each type of physics.')
                # Create a dicitionary entry for the tag with the key as the tag and
                # the value as the physics_id
                self.tag[physics.tag] = physics_id
                physics_id += 1
    
    def set_element(self):
        """
        Set the finite element for the multiphysics problem by creating a mixed element
        from the elements of the subphysics problems.
        """

        self.sub_elements = []
        # Create a list of subphysics elements
        for physics in self.sub_physics:
            self.sub_elements.append(physics.element)

        # Make the monolithic mixed element from the subphysics elements
        self.element = basix.ufl.mixed_element(self.sub_elements) 

    def set_function_space(self, V):
        """
        Set the function space for the multiphysics problem and for each subphysics problem.
        Parameters
        ----------
        V : dolfinx.fem.FunctionSpace
            The function space for the multiphysics problem.
        """

        self.V = V
        for physics_id, physics in enumerate(self.sub_physics):
            # Set the function space for each subphysics problem
            # Not recursive - physics.set_function_space calls sub physics set_function_space
            physics.set_function_space(self.V.sub(physics_id))

    def set_test_function(self, multiphysics_test_function):
        """
        Set the test function for the multiphysics problem and for each subphysics problem.
        Parameters
        ----------
        multiphysics_test_function : ufl.TestFunction
            The test function for the multiphysics problem. 
        """

        self.test_function = multiphysics_test_function
        te_split = ufl.split(self.test_function)
        for physics_id, physics in enumerate(self.sub_physics):
            # Set the test function for each subphysics problem
            # Not recursive 
            physics.set_test_function(te_split[physics_id])

    def set_trial_function(self, multiphysics_trial_function):
        """
        Set the trial function for the multiphysics problem and for each subphysics problem.
        Parameters
        ----------
        multiphysics_trial_function : ufl.TrialFunction
            The trial function for the multiphysics problem.
        """

        self.trial_function = multiphysics_trial_function
        tr_split = ufl.split(self.trial_function)
        for physics_id, physics in enumerate(self.sub_physics):
            # Set the trial function for each subphysics problem
            # Not recursive 
            physics.set_trial_function(tr_split[physics_id])
    
    def set_solution_function(self, multiphysics_solution_function):
        """
        Set the solution function for the multiphysics problem and for each subphysics problem.
        Parameters
        ----------
        multiphysics_solution_function : dolfinx.fem.Function
            The solution function for the multiphysics problem.
        """
        self.solution = multiphysics_solution_function
        sol_split = ufl.split(self.solution)
        for physics_id, physics in enumerate(self.sub_physics):
            # Set the solution function for each subphysics problem
            # Not recursive 
            physics.set_solution_function(sol_split[physics_id])

    def get_function_space(self, physics_tag=None):
        """
        Get the function space for a given physics tag.
        Parameters
        ----------
        physics_tag : str, optional
            The tag of the physics problem. If None, returns the function space for the multiphysics problem.
        Returns
        -------
        dolfinx.fem.FunctionSpace
            The function space associated with the given physics tag, or the multiphysics function space if no tag is provided.
        """
        if physics_tag is None:
            return self.V
        return self.get_physics(physics_tag).get_function_space(physics_tag)
    
    def get_test_function(self, physics_tag=None):
        """
        Get the test function for a given physics tag.
        Parameters
        ----------
        physics_tag : str, optional
            The tag of the physics problem. If None, returns the test function for the multiphysics problem.
        Returns
        -------
        ufl.core.expr.Expr
            The test function associated with the given physics tag, or the multiphysics test function if no tag is provided.
        """
        
        if physics_tag is None:
            return self.test_function
        return self.get_physics(physics_tag).get_test_function(physics_tag)
        
    def get_trial_function(self, physics_tag=None):
        """
        Get the trial function for a given physics tag.
        
        Parameters
        ----------
        physics_tag : str, optional
            The tag of the physics problem. If None, returns the trial function for the multiphysics problem.
        Returns
        -------
        ufl.core.expr.Expr
            The trial function associated with the given physics tag, or the multiphysics trial function if no tag is provided.
        """
        if physics_tag is None:
            return self.trial_function
        return self.get_physics(physics_tag).get_trial_function(physics_tag)
    
    def get_solution_function(self, physics_tag=None):
        """
        Get the solution function for a given physics tag.
        
        Parameters
        ----------
        physics_tag : str, optional
            The tag of the physics problem. If None, returns the solution function for the multiphysics problem.
        Returns
        -------
        ufl.core.expr.Expr
            The solution function associated with the given physics tag, or the multiphysics solution function if no tag is provided.   
        """
        # If no physics tag is provided, return the solution function for the multiphysics problem
        if physics_tag is None:
            return self.solution
        
        # If a physics tag is provided, get the physics ID and return the solution function for that physics problem
        physics_id = self.get_physics_id(physics_tag)

        # If the physics problem is a mixed element, return the solution function for that specific tag
        # Otherwise, return the solution function for the multiphysics problem
        if self._is_mixed_element[physics_id]:
            return self.sub_physics[physics_id].get_solution_function(physics_tag)
        
        return self.sub_physics[physics_id].get_solution_function()
    
    def get_physics(self, physics_tag):
        """
        Get the physics problem for a given physics tag.
        
        Parameters
        ----------
        physics_tag : str
            The tag of the physics problem.
        Returns
        -------
        PhysicsProblem
            The physics problem associated with the given tag.
        """
        physics_id = self.get_physics_id(physics_tag)
        return self.sub_physics[physics_id]

    def get_physics_id(self, physics_tag):
        """
        Get the physics ID for a given physics tag.
        
        Parameters
        ----------
        physics_tag : str
            The tag of the physics problem.
        Returns
        -------
        int
            The ID of the physics problem.
        """
        return self.tag[physics_tag]
    
    def get_global_dofs(self, global_function_space, physics_tag=None, sort=True):
        """
        Get the global degrees of freedom (DOFs) indices on the local process for a given physics tag or 
        for all physics problems.
        
        Parameters
        ----------
        global_function_space : dolfinx.fem.FunctionSpace
            The global function space from which to extract the DOFs.
        physics_tag : str or list of str, optional
            The tag(s) of the physics problem(s). If None, returns the DOFs for all physics problems.
        sort : bool, optional
            Whether to sort the DOFs before returning. Default is True.
        
        Returns
        -------
        np.ndarray
            An array of global DOFs indices on local process associated with the given physics tag(s), or all DOFs
        """

        W = global_function_space
        
        # Get all physics tags in the multiphysics system
        if physics_tag is None:
            physics_tag = self.tag.keys()

        # If the physics tag is not a list, we wrap it in a list to reuse the same logic
        if not _is_container(physics_tag):
            physics_tag = [physics_tag]

        dofs = []
        for tag in physics_tag:
            V = self.get_function_space(tag)
            V, map_WV = V.collapse()

            # Get the local dofs for the function space and strip the ghost nodes
            local_V = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
            map_VW_stripped = np.asarray(map_WV)[:local_V]
            V_dofs_global = W.dofmap.index_map.local_to_global(map_VW_stripped)

            dofs.append(V_dofs_global)

        dofs = np.concatenate(dofs)

        if sort:
            dofs = np.sort(dofs)

        return np.array(dofs)
        
    def set_weak_form(self, *options):
        """
        Set the weak form for the multiphysics problem by summing the weak forms of each subphysics problem.
        
        Parameters
        ----------
        options : dict, optional
            Options for each subphysics problem's weak form. If not provided, 
            an empty dictionary is used for each subphysics problem.
        
        Raises
        ------
        ValueError
            If a subphysics problem does not have the attribute weak_form defined. 
        """
        options = list(options)

        # Add empty dictionary as options for each physics if the option is not provided.
        while len(options) < self.num_sub_physics:
            options.append({})

        weak_forms = []
        for i in range(self.num_sub_physics):
            self.sub_physics[i].set_weak_form(**options[i])
            
            if hasattr(self.sub_physics[i], 'weak_form'):
                weak_forms.append(self.sub_physics[i].weak_form)

            else:
                raise ValueError(f'Physics problem {i} does not have attribute weak_form defined. \
                                 Please define a weak form for each physics problem before setting the multiphysics weak form.')
            
            self.weak_form = sum(weak_forms)

    def flux(self, h, physics_tag):
        """
        Get the flux for a given physics tag.
        
        Parameters
        ----------
        h : ufl.core.expr.Expr
            The test function for the flux calculation.
        physics_tag : str
            The tag of the physics problem for which to calculate the flux.
        Returns
        -------
        ufl.core.expr.Expr
            The flux expression for the specified physics problem.
        """
        physics_id = self.get_physics_id(physics_tag)
        return self.sub_physics[physics_id].flux(h)
    
    def get_residual(self):
        """
        Get the residual of the multiphysics problem by summing the residuals of each subphysics problem.
        
        Returns
        -------
        ufl.core.expr.Expr
            The residual expression for the multiphysics problem.
        """
        
        residual = self.sub_physics[0].get_residual()
        for physics in self.sub_physics[1:]:
            residual += physics.get_residual()
        return residual
    
    def set_bcs(self, multiphysics_bc_dict):
        """
        Set the boundary conditions for the multiphysics problem by iterating over the boundary conditions of each subphysics problem.

        Parameters
        ----------
        multiphysics_bc_dict : dict
            A dictionary containing the boundary conditions for each subphysics problem. 
            The keys are the boundary IDs and the values are dictionaries with the boundary condition type and value.
        
        Raises
        ------
        ValueError
            If a boundary condition type is not recognized or if the physics ID is not found in the multiphysics problem.
        
        """
        
        sub_bcs = [{} for i in range(self.num_sub_physics)]
        
        for physics_tag in multiphysics_bc_dict:
            bc_dict = multiphysics_bc_dict[physics_tag]
            physics_id = self.get_physics_id(physics_tag)            

            for boundary_id in bc_dict:
                bc_data = bc_dict[boundary_id]
                bc_type = bc_data['type']
                
                if bc_type == 'dirichlet':
                    # Handles multiphyics problems inside multiphysics problems
                    if self._is_mixed_element[physics_id]:
                        if physics_tag not in sub_bcs[physics_id]:
                            sub_bcs[physics_id][physics_tag] = {}
                        sub_bcs[physics_id][physics_tag][boundary_id] = bc_data

                    else:
                        sub_bcs[physics_id][boundary_id] = bc_data

                elif bc_type == 'neumann':
                    bc_value = bc_data['value']
                    flux_term = self.sub_physics[physics_id].flux(bc_value)
                    self.add_to_weak_form(flux_term, self.ds(boundary_id))

        # Set the boundary conditions for each subphysics problem
        for physics_id, sub_bc in enumerate(sub_bcs):            
            self.sub_physics[physics_id].set_bcs(sub_bc)
            
        # Add the Dirichlet boundary conditions from each subphysics problem to the multiphysics problem
        for physics_id in range(self.num_sub_physics):
            self.dirichlet_bcs.extend(self.sub_physics[physics_id].dirichlet_bcs)

    def set_writer(self, output_dir, file_format):
        """
        Set the writer for each subphysics problem to write output to the specified directory.
        
        Parameters
        ----------
        output_dir : str
            The directory where the output files will be written.
        file_format : str, optional
            The file format for the output files.
        
        """
        
        if MPI.COMM_WORLD.rank == 0:
            # Remove the output directory if it already exists
            if os.path.exists(output_dir):
                subprocess.run(['rm', '-rf', output_dir])
        
            subprocess.run(['mkdir', '-p', output_dir])
        MPI.COMM_WORLD.barrier()

        for physics_id, physics in enumerate(self.sub_physics):
            physics.set_writer(output_dir, file_format)

    def write_subphysics(self, physics, solution, *args, **kwargs):
        """
        Recursive function that splits (monolithic) multiphysics problems and solutions 
        until each sub solution can be written.

        Example
        ----------
        ::

            Coupled-Flow-Transport[Navier-Stokes[momentum, continuity], Transport]
                |
            Split 1
                |----Navier-Stokes[momentum, continuity] (call function)
                |       |
                |    Split 2
                |       |----Momentum -> write to file
                |       |
                |       |----Continuity -> write to file
                |
                |----Transport -> write to file 
        

        Parameters
        ----------
        physics : PhysicsProblem
            The physics problem to write the solution for.
        solution : dolfinx.fem.Function
            The solution function to write.
        
        """
        if hasattr(physics, 'sub_physics'):
            sub_solutions = solution.split()
            for physics_id, sub_physics in enumerate(physics.sub_physics):
                self.write_subphysics(sub_physics, sub_solutions[physics_id], *args, **kwargs)

        else:
            solution = solution.collapse()
            solution.name = physics.tag
            physics.write(solution, *args, **kwargs)

    def write(self, function_to_save=None, *args, **kwargs):
        """
        Write the solution of the multiphysics problem to files.
        Parameters
        ----------
        function_to_save : ufl.core.expr.Expr, optional
            The function to save. If None, the solution function of the multiphysics problem is used.
        *args : tuple
            Additional positional arguments to pass to the write method of each subphysics problem.
        **kwargs : dict
            Additional keyword arguments to pass to the write method of each subphysics problem.
        
        """

        if function_to_save is None:
            sols = self.get_solution_function()
        else:
            sols = function_to_save

        sols_components = sols.split()

        for physics_id, sub_physics in enumerate(self.sub_physics):
            self.write_subphysics(sub_physics, sols_components[physics_id], *args, **kwargs)

class TransientMultiPhysicsProblem(MultiphysicsProblem):
    """
    Extension of MultiphysicsProblem to support transient simulations.

    This class manages the storage and update of the previous time step's solution, which 
    is essential for time-stepping schemes in transient simulations involving multiple 
    coupled physics components.

    """

    def build_function_space(self):
        """
        Build the function space for the multiphysics problem and initialize the previous solution function.
        This method extends the `build_function_space` method of the `MultiphysicsProblem` class
        by also creating a function to store the solution from the previous time step.
        
        """
        
        super().build_function_space()
        self.previous_solution = dolfinx.fem.Function(self.get_function_space())
        prev_sol_split = ufl.split(self.previous_solution)

        for physics_id, physics in enumerate(self.sub_physics): 
            physics.previous_solution = prev_sol_split[physics_id]

    def update_previous_solution(self):
        """
        Update the previous solution function with the current solution.
        This method copies the values from the current solution function to the previous solution function, 
        preparing it for the next time step in a transient simulation.
        
        """
        self.previous_solution.x.array[:] = self.solution.x.array[:]



