import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import copy
import os
from collections.abc import Iterable

# ------------------------------------------------------- #

from ..info.messages import import_fenics
fe = import_fenics()

from .physics_problem import PhysicsProblem

def _split_element(element):
    """
    if element is a single FiniteElement, VectorElement, or TensorElement, return the element
    if the element is a mixed element, return a list of the subelements
    """
    if element.shortstr().startswith('Mixed'):
        return element.sub_elements()
    else:
        return element

def _is_container(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))

class MultiPhysicsProblem(PhysicsProblem):

    """
    A class representing a multi-physics problem composed of multiple individual physics problems. The PhysicsProblems
    must be defined on the same mesh space

    :param physics_problems: PhysicsProblem objects or their derived classes. You can initialize this class with as many PhysicsProblem object as needed. It is noted that every PhysicsProblem **must** have a unique tag.

    """

    def __init__(self, *physics_problems):

        # Verify that all of the problems have the same mesh
        mesh_hash = physics_problems[0].mesh.fenics_mesh().hash()
        for physics in physics_problems:
            if physics.mesh.fenics_mesh().hash() != mesh_hash:
                raise ValueError("Mesh in the physics problems are not the same")

        # Put the subphysics in a list
        self.sub_physics = physics_problems
        self.num_sub_physics = len(self.sub_physics)

        # Since all of the physics problem have the same mesh, I just grab the first mesh
        # to initialize my physics problem
        super().__init__(physics_problems[0].mesh)

        # Set problem tags as a dictionary to the physics id
        self.tag = {}
        self._is_mixed_element = []
        physics_id = 0

        for physics in self.sub_physics:

            if isinstance(physics.tag, dict):
                self._is_mixed_element.append(True)
                for tag in physics.tag:

                    if tag in self.tag.keys():
                        raise ValueError("Duplicate physics tag detected in a MultiPhysicsProblem."
                                " Please set a unique tag for each type of physics.")

                    self.tag[tag] = physics_id

                physics_id += 1

            else:
                self._is_mixed_element.append(False)
                if physics.tag in self.tag.keys():
                    raise ValueError("Duplicate physics tag detected in a MultiPhysicsProblem."
                            " Please set a unique tag for each type of physics.")

                self.tag[physics.tag] = physics_id
                physics_id += 1

    def set_element(self):
        """
        Build FEniCS MixedElement from the list of sub elements
        """
        self.sub_elements = []
        for physics in self.sub_physics:
            self.sub_elements.append(physics.element)
        self.element = fe.MixedElement(self.sub_elements)

    def set_function_space(self, V):
        self.V = V
        for physics_id, physics in enumerate(self.sub_physics):
            physics.set_function_space(self.V.sub(physics_id))

    def set_test_function(self, te):
        self.test_func = te
        te_split = fe.split(self.test_func)
        for physics_id, physics in enumerate(self.sub_physics):
            physics.set_test_function(te_split[physics_id])

    def set_trial_function(self, tr):
        self.trial_func = tr
        tr_split = fe.split(self.trial_func)
        for physics_id, physics in enumerate(self.sub_physics):
            physics.set_trial_function(tr_split[physics_id])

    def set_solution_function(self, sol):
        self.solution = sol
        sol_split = fe.split(self.solution)
        for physics_id, physics in enumerate(self.sub_physics):
            physics.set_solution_function(sol_split[physics_id])

    def function_space(self, physics_tag=None):
        if physics_tag is None:
            return self.V
        return self.get_physics(physics_tag).function_space(physics_tag)

    def test_function(self, physics_tag=None):
        if physics_tag is None:
            return self.test_func
        return self.get_physics(physics_tag).test_function(physics_tag)

    def trial_function(self, physics_tag=None):
        if physics_tag is None:
            return self.trial_func
        return self.get_physics(physics_tag).trial_function(physics_tag)

    def solution_function(self, physics_tag=None):

        """
        If physics_tag is None, this function will return the fe.Function object containing every solution field.
        If physics_tag is provide, this function will provide a reference to the specific solution field inside the full fe.Function object
        """

        if physics_tag is None:
            return self.solution

        physics_id = self.get_physics_id(physics_tag)
        if self._is_mixed_element[physics_id]:
            return self.get_physics(physics_tag).solution_function(physics_tag)
        return self.get_physics(physics_tag).solution_function()

    def get_physics_id(self, physics_tag):
        """
        Get the id order of the physics based on the tag
        """
        return self.tag[physics_tag]

    def get_physics(self, physics_tag):
        """
        Get the underlying physics object
        """
        physics_id = self.get_physics_id(physics_tag)
        return self.sub_physics[physics_id]

    def get_dofs(self, physics_tag=None, sort=True):

        """
        Return the global dofs ids for a the physics tag
        if physics_tag is None, return the dofs of this entire multiphysics
        """

        if not _is_container(physics_tag):
            V = self.function_space(physics_tag)
            return V.dofmap().dofs()

        # If dofs is a tuple containing multiple tags,
        # get the dofs from the individual tags, and 
        # stack them together in an ascending manner
        dofs = []
        for tag in physics_tag:
            dofs += self.function_space(tag).dofmap().dofs()
        if sort: dofs.sort()
        return dofs

    def set_weak_form(self, *options):

        """
        The weak form here is the sum of all of the weak forms from every physics
        """

        # Add empty dictionary as options for each physics if the option is not provided.
        options = list(options)
        while len(options) < self.num_sub_physics:
            options.append({})

        weak_forms = []
        for i in range(self.num_sub_physics):
            self.sub_physics[i].set_weak_form(**options[i])
            if hasattr(self.sub_physics[i], "weak_form"):
                weak_forms.append(self.sub_physics[i].weak_form)
        self.weak_form = sum(weak_forms)

    def flux(self, h, physics_tag):
        physics_id = self.get_physics_id[physics_tag]
        return self.sub_physics[physics_id].flux(h)

    def get_residue(self):
        residue = self.sub_physics[0].get_residue()
        for i in range(1, self.num_sub_physics):
            residue += self.sub_physics[i].get_residue()
        return residue

    def set_bcs(self, multiphysics_bcs_dict):

        sub_bcs = [{} for i in range(self.num_sub_physics)]

        for physics_tag in multiphysics_bcs_dict:

            bcs_dict = multiphysics_bcs_dict[physics_tag]
            physics_id = self.get_physics_id(physics_tag)

            for boundary_id in bcs_dict:
                bc_data = bcs_dict[boundary_id]
                bc_type = bc_data['type']

                if bc_type == 'dirichlet':
                    if self._is_mixed_element[physics_id]:
                        if physics_tag not in sub_bcs[physics_id]:
                            sub_bcs[physics_id][physics_tag] = {}
                        sub_bcs[physics_id][physics_tag][boundary_id] = bc_data
                    else:
                        sub_bcs[physics_id][boundary_id] = bc_data

                elif bc_type == 'neumann':
                    # Neumann is added manually here because superclass set_bcs adds the term to the local weak form
                    bc_value = self.sub_physics[physics_id].get_zero_constant() if bc_data['value'] == 'zero' else bc_data['value']
                    flux_term = self.sub_physics[physics_id].flux(bc_value)
                    self.add_to_weakform(flux_term, self.ds(boundary_id))

        # Set dirichlet BC for each sub physics
        for physics_id, sub_bc in enumerate(sub_bcs):
            self.sub_physics[physics_id].set_bcs(sub_bc)

        # Get the subphysics bcs list and add it to the multiphysics bcs
        for physics_id in range(self.num_sub_physics):
            self.dirichlet_bcs.extend(self.sub_physics[physics_id].dirichlet_bcs)

    def set_writer(self, directory, file_format):
        os.system("mkdir -p %s" % directory)
        for physics_id, physics in enumerate(self.sub_physics):
            physics.set_writer(directory, file_format)

    def write(self, function_to_save=None, *args, **kwargs):
        if function_to_save is None:
            sol = self.solution_function()
        else:
            sol = function_to_save
        sols = sol.split(deepcopy=True)
        for physics_id, physics in enumerate(self.sub_physics):
            physics.write(function_to_save=sols[physics_id], *args, **kwargs)


class TransientMultiPhysicsProblem(MultiPhysicsProblem):

    def build_function_space(self):

        super().build_function_space()
        self.previous_solution = fe.Function(self.function_space())
        prev_sol_split = fe.split(self.previous_solution)
        for physics_id, physics in enumerate(self.sub_physics):
            physics.previous_solution = prev_sol_split[physics_id]

    def update_previous_solution(self):
        self.previous_solution.assign(self.solution_function())

