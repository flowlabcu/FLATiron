import numpy as np

from flatiron_tk.info import *
adios4dolfinx = import_adios4dolfinx()
basix = import_basix()
dolfinx = import_dolfinx()
PETSc = import_PETSc()
ufl = import_ufl()

from collections.abc import Iterable
from flatiron_tk.solver import ConvergenceMonitor
from flatiron_tk.solver import NonLinearSolver


"""
This function builds a block preconditioner P for ksp solve of matrix A.

Let ksp(A, P) means a Krylov solve of matrix A with preconditioner P

Define a block matrix system A as

    A = [ A00, A01
          A10, A11 ]

Composite type options are:

    additive: P = [ksp(A00,Ap00), 0
                   0            , ksp(A11,Ap11)]

    multiplicative: P = J.K.L
        where J = [I, 0
                   0, ksp(A11,Ap11)]

              K = [0, 0    +   [ I  , 0     *[I, 0        (see PETSc's doc page for details here)
                   0, I]        -A10, -A11]   0, 0]

              L = [ksp(A00,Ap00), 0
                   0            , I]

    symmetric_multiplicative: (see PETSc's doc page. This is too big)

    schur: (schur complement decomposition, see doc page)
"""

def _is_container(obj):
    """
    Utility function to check if an object is a container.

    Parameters:
    ------------
        obj: The object to check.
    
    Returns:
    ------------ 
        bool: True if the object is a container (like a list, tuple, set),
              False otherwise.
    """
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))

class BlockSplitNode():
    """
    A node in the block split tree for managing block preconditioners.
    """
    def __init__(self):
        # Default the node to not be a root node
        self._is_root = False
        self.left_node = None
        self.right_node = None
        self.parent_node = None
        pass
    
    def set_root_function_space(self, V):
        """
        Set the root function space for this node.
        
        Parameters:
        ------------
            V (dolfinx.fem.FunctionSpace): The function space to set as the root.
        """
        self.root_function_space = V

    def set_field_tags(self, fields):
        """
        Set the field names for this node. Each field name is equivalent to the
        tag of a PhysicsProblem in the flatiron_tk framework

        Parameters:
        ------------
            fields (Iterable[str]): An iterable of field tags (physics tags) to set.

        Sets: 
        ------------
            _field_tags (tuple): A tuple of field names.
            _node_tag (str): A string that concatenates the field names with an underscore.
        """
        
        self._field_tags = tuple(sorted(fields))
        self._node_tag = '_'.join(self._field_tags)

    def set_as_root(self):
        """
        Set this node as the root node of the block split tree.
        """
        self._is_root = True

    def set_ksp(self, ksp):
        """
        Set the KSP (Krylov Subspace Solver) for this node.

        Parameters:
        ------------
            ksp (PETSc.KSP): The KSP object to set.
            _IS (Optional[PETSc.IS]): Optional PETSc Index Set for the node.
        """
        self._ksp = ksp

    def set_IS(self, IS):
        """
        Set the PETSc Index Set for this node.

        Parameters:
        ------------
            IS (PETSc.IS): The Index Set to set.
        """
        self._IS = IS

    def insert_node(self, child_node, position):
        """
        Insert a child node at the specified position ('left' or 'right').

        Parameters:
        ------------
            node (BlockSplitNode): The node to insert.
            position (str): The position to insert the node ('left' or 'right').
        """
        if position == 'left':
            # Ensure the node is not already set
            assert self.left_node is None, "Left node already set."
            self.left_node = child_node
            
        elif position == 'right':
            # Ensure the node is not already set
            assert self.right_node is None, "Right node already set."
            self.right_node = child_node
            
    # ---- Getters ---- #
    def is_root(self):
        """
        Check if this node is the root node.
        
        Returns:
        ------------
            bool: True if this node is the root, False otherwise.
        """
        return self._is_root
    
    def get_field_tags(self):
        """
        Get the field tags associated with this node.
        
        Returns:
        ------------
            tuple: A tuple of field tags.
        """
        return self._field_tags
    
    def get_ksp(self):
        """
        Get the KSP (Krylov Subspace Solver) associated with this node.
        
        Returns:
        ------------
            PETSc.KSP: The KSP object for this node.
        """
        return self._ksp
        
    def get_node_tag(self):
        """
        Get the tag of this node, which is a string concatenation of field tags.
        
        Returns:
        ------------
            str: The node tag.
        """
        return self._node_tag
    
class BlockSplitTree():
    """
    A class to manage a block split tree for building block preconditioners.

    Parameters
    ------------
        physics (PhysicsProblem): The physics problem containing field tags and function spaces.
        splits (dict or Iterable[dict]): A dictionary or an iterable of dictionaries defining the block
            splits. Each dictionary should have the following keys:

            - 'fields': A list of two lists, each containing field tags to be grouped together.
            - 'composite_type': The type of composite preconditioner to use ('additive', 'multiplicative',
                'symmetric_multiplicative', 'schur', 'special').
            - 'schur_fact_type' (optional): The Schur factorization type if 'composite_type' is 'schur'
                ('diag', 'full', 'lower', 'upper').
            - 'schur_pre_type' (optional): The Schur preconditioner type if 'composite_type' is 'schur'
                ('a11', 'full', 'self', 'selfp', 'user').
            - 'ksp0_set_function' (optional): A function to set up the KSP for the first block.
            - 'ksp1_set_function' (optional): A function to set up the KSP for the second block.    
    """
    def __init__(self, physics, splits):
        self.physics = physics

        # Set the root node of the tree
        self.root = BlockSplitNode()
        self.root.set_as_root()
        self.root.set_field_tags(list(physics.tag.keys()))
        self.root.set_root_function_space(physics.get_function_space())

        # Set the dictionary mapping field tags to nodes
        self.node_dict = {self.root.get_field_tags(): self.root}
        
        # Store the splits as a list of dictionaries. 
        # If a single split is provided, convert it to a list
        # If a list or other iterable is provided, use it directly
        if isinstance(splits, dict): self.splits = [splits]
        elif _is_container(splits): self.splits = splits
        else: raise ValueError("Splits must be a dictionary or an iterable of dictionaries.")
        
        # Set the supported PETSc blocksplit dictionaries. Separate method for clarity
        self._set_PETSc_fieldsplit_dictionary()
    
    def _set_PETSc_fieldsplit_dictionary(self):
        """
        Set the PETSc fieldsplit dictionary for the block split tree.

        Supported composite types:
        
        - 'additive'
        - 'multiplicative'
        - 'symmetric_multiplicative'
        - 'schur'
        - 'special'

        Supported schur factorization types:
        
        - 'diag'
        - 'full'
        - 'lower'
        - 'upper'

        Supported schur preconditioner types:
        
        - 'a11'
        - 'full'
        - 'self'
        - 'selfp'
        - 'user'
        """

        # block split composite type. This tells you what the final
        # preconditioner look like
        # See https://petsc.org/release/manual/ksp/#sec-block-matrices
        self._composite_type_dict = {}
        self._composite_type_dict['additive'] = PETSc.PC.CompositeType.ADDITIVE
        self._composite_type_dict['multiplicative'] = PETSc.PC.CompositeType.MULTIPLICATIVE
        self._composite_type_dict['schur'] = PETSc.PC.CompositeType.SCHUR
        self._composite_type_dict['special'] = PETSc.PC.CompositeType.SPECIAL
        self._composite_type_dict['symmetric_multiplicative'] = PETSc.PC.CompositeType.SYMMETRIC_MULTIPLICATIVE

        # Schur complement factorization type
        # If composite_type is SCHUR, this tells you
        # what the final preconditioner involving the schur complement
        # look like.
        # See: https://petsc.org/release/manualpages/PC/PCFieldSplitSetSchurPre/
        self._schur_fact_type_dict = {}
        self._schur_fact_type_dict['diag'] = PETSc.PC.SchurFactType.DIAG
        self._schur_fact_type_dict['full'] = PETSc.PC.SchurFactType.FULL
        self._schur_fact_type_dict['lower'] = PETSc.PC.SchurFactType.LOWER
        self._schur_fact_type_dict['upper'] = PETSc.PC.SchurFactType.UPPER

        # Schur complement preconditioner type
        # Preconditioner for the schur complement block
        # See https://petsc.org/release/manualpages/PC/PCFieldSplitSetSchurPre/
        self._schur_pre_type_dict = {}
        self._schur_pre_type_dict['a11'] = PETSc.PC.SchurPreType.A11
        self._schur_pre_type_dict['full'] = PETSc.PC.SchurPreType.FULL
        self._schur_pre_type_dict['self'] = PETSc.PC.SchurPreType.SELF
        self._schur_pre_type_dict['selfp'] = PETSc.PC.SchurPreType.SELFP
        self._schur_pre_type_dict['user'] = PETSc.PC.SchurPreType.USER

    def split_IS(self, blocks_0, blocks_1):
        """
        Split the blocks into two sets and create a PETSc Index Set (IS) for each.
        
        Parameters:
        ------------
            blocks_0 (Iterable[str]): The first set of blocks (field tags).
            blocks_1 (Iterable[str]): The second set of blocks (field tags).

        Returns:
        ------------
            tuple: A tuple containing two PETSc Index Sets (IS) for the blocks.
        """
        _blocks_0, _blocks_1 = blocks_0, blocks_1
        # Ensure blocks are iterable
        if not _is_container(_blocks_0):
            _blocks_0 = [_blocks_0]
        if not _is_container(_blocks_1):
            _blocks_1 = [_blocks_1]

        # Make sure all blocks in the parent node are in the child nodes 
        parent_blocks = tuple(sorted(list(_blocks_0) + list(_blocks_1)))        
        assert parent_blocks in self.node_dict
        parent_node = self.node_dict[parent_blocks]

        # Set the block names as a combination of the field tags
        blocks_0_name = '_'.join(list(_blocks_0))
        blocks_1_name = '_'.join(list(_blocks_1))

        # Create a new node for the split (left)
        left_node = BlockSplitNode()
        left_node.set_field_tags(_blocks_0)
        parent_node.insert_node(left_node, 'left')
        self.node_dict[left_node.get_field_tags()] = left_node

        # Create a new node for the split (right)
        right_node = BlockSplitNode()
        right_node.set_field_tags(_blocks_1)
        parent_node.insert_node(right_node, 'right')
        self.node_dict[right_node.get_field_tags()] = right_node

        # Build the PETSc Index Sets (IS) for the blocks
        dofs0 = self.physics.get_global_dofs(
            self.root.root_function_space,
            parent_node.left_node.get_field_tags(), 
            sort=False
        )
        
        dofs1 = self.physics.get_global_dofs(
            self.root.root_function_space,
            parent_node.right_node.get_field_tags(), 
            sort=False)

        is0, is1 = self.get_block_split_index_set(parent_node, dofs0, dofs1)

        return [blocks_0_name, is0], [blocks_1_name, is1]

    def build_block_split_pc(self, blocks_0, blocks_1, 
                             is0_data, is1_data, 
                             composite_type='additive',
                             schur_fact_type='full',
                             schur_pre_type='a11'):
        
        """
        Build the block split preconditioner for the given blocks.
        
        Parameters
        ------------
            blocks_0 (Iterable[str]): The first set of blocks (field tags).
            blocks_1 (Iterable[str]): The second set of blocks (field tags).
            is0_data (tuple): A tuple containing the name and PETSc Index Set for the first block.
            is1_data (tuple): A tuple containing the name and PETSc Index Set for the second block.
            composite_type (str): The type of composite preconditioner to use.

                Options: 'additive', 'multiplicative', 'symmetric_multiplicative',
                'schur', 'special'. Default is 'additive'.
            
            schur_fact_type (str): The Schur factorization type to use if composite_type
                is 'schur'. Options: 'diag', 'full', 'lower', 'upper'. Default is 'full'.
            
            schur_pre_type (str): The Schur preconditioner type to use if composite_type
                is 'schur'. Options: 'a11', 'full', 'self', 'selfp', 'user'. Default is 'a11'. 
        
        Returns
        ------------
            PETSc.KSP: The KSP object for the block split preconditioner.
        """
        _blocks_0, _blocks_1 = blocks_0, blocks_1
        # Ensure blocks are iterable
        if not _is_container(_blocks_0):
            _blocks_0 = [_blocks_0]
        if not _is_container(_blocks_1):
            _blocks_1 = [_blocks_1]

        # Make sure all blocks in the parent node are in the child nodes 
        parent_blocks = tuple(sorted(list(_blocks_0) + list(_blocks_1)))
        assert parent_blocks in self.node_dict
        parent_node = self.node_dict[parent_blocks]

        # Get the KSP from the parent node and set the preconditioner to fieldsplit
        parent_ksp = parent_node.get_ksp()
        pc = parent_ksp.pc
        pc.setType(PETSc.PC.Type.FIELDSPLIT)
        pc.setFieldSplitIS(is0_data, is1_data)

        # Ensure the composite type is valid
        assert composite_type in self._composite_type_dict
        comp_type = self._composite_type_dict[composite_type]
        pc.setFieldSplitType(comp_type)

        if composite_type == 'schur':
            # If the composite type is 'schur', set the Schur factorization type
            assert schur_pre_type in self._schur_pre_type_dict
            pre_type = self._schur_pre_type_dict[schur_pre_type]
            pc.setFieldSplitSchurPreType(pre_type)

            assert schur_fact_type in self._schur_fact_type_dict
            fact_type = self._schur_fact_type_dict[schur_fact_type]
            pc.setFieldSplitSchurFactType(fact_type)

        parent_ksp.setUp()
        return pc.getFieldSplitSubKSP()
         
    def get_block_split_index_set(self, node, dofs0, dofs1):
        """
        Get the PETSc Index Set (IS) for the block split.

        Parameters:
        ------------
            node (BlockSplitNode): The node for which to get the IS.
            dofs0 (np.ndarray): DOFs for the first block.
            dofs1 (np.ndarray): DOFs for the second block.

        Returns:
        ------------
            PETSc.IS: The Index Set for the block split.
        """
        comm = self.physics.mesh.comm

        # Root node: dofs are already monolithic global indices (owned)
        if node.is_root():
            is0 = PETSc.IS().createGeneral(np.sort(dofs0).astype(np.int32), comm=comm)
            is1 = PETSc.IS().createGeneral(np.sort(dofs1).astype(np.int32), comm=comm)
            return is0, is1
        
        # If the node is not the root, create separate IS for each block
        # Grab the communicator from the physics mesh
        comm = self.physics.mesh.comm

        # Sort the DOFs, find the length of the arrays
        num_dofs0 = len(dofs0)
        num_dofs1 = len(dofs1)
        num_dofs_sub = num_dofs0 + num_dofs1

        node_dofs = np.concatenate((dofs0, dofs1)).astype(np.int32)
        node_dofs_sorted = np.sort(node_dofs)

        # Find the offset for parallel DOFs
        sub_ids_0 = np.where(node_dofs_sorted < num_dofs0)[0].astype(np.int32)
        sub_ids_1 = np.where(node_dofs_sorted >= num_dofs0)[0].astype(np.int32)
        offset = np.cumsum([0] + comm.allgather(num_dofs_sub))[comm.rank]

        sub_ids_0 += offset
        sub_ids_1 += offset

        # Create the PETSc Index Sets for the sub-blocks
        is0 = PETSc.IS().createGeneral(np.sort(sub_ids_0).astype(np.int32))
        is1 = PETSc.IS().createGeneral(np.sort(sub_ids_1).astype(np.int32))

        return is0, is1
    
    def set_child_node_ksps(self, blocks_0, blocks_1, is0_data, is1_data, ksp0, ksp1):
        """
        Set the KSPs for the child nodes of the block split.
        Parameters
        ------------
            blocks_0 (Iterable[str]): The first set of blocks (field tags).
            blocks_1 (Iterable[str]): The second set of blocks (field tags).
            is0_data (tuple): A tuple containing the name and PETSc Index Set for the first block.
            is1_data (tuple): A tuple containing the name and PETSc Index Set for the second block.
            ksp0 (PETSc.KSP): The KSP for the first block.
            ksp1 (PETSc.KSP): The KSP for the second block.
        
        Returns
        ------------
            None
        """
        _blocks_0, _blocks_1 = blocks_0, blocks_1
        # Ensure blocks are iterable
        if not _is_container(_blocks_0):
            _blocks_0 = [_blocks_0]
        if not _is_container(_blocks_1):
            _blocks_1 = [_blocks_1]

        # Make sure all blocks in the parent node are in the child nodes 
        parent_blocks = tuple(sorted(list(_blocks_0) + list(_blocks_1)))
        assert parent_blocks in self.node_dict
        parent_node = self.node_dict[parent_blocks]

        # Insert the KSPs into the child nodes
        parent_node.left_node.set_ksp(ksp0)
        parent_node.right_node.set_ksp(ksp1)
        parent_node.left_node.set_IS(is0_data[1])
        parent_node.right_node.set_IS(is1_data[1])
        
class BlockNonLinearSolver(NonLinearSolver):
    def __init__(self, block_split_tree: BlockSplitTree, *args, **kwargs):
        self._block_split_tree = block_split_tree    
        super().__init__(*args, **kwargs) 

    def init_ksp(self):
        """
        Override the init_ksp method to set up the KSP solver
        according to the block split tree configuration.
        """
        if self.ksp_is_initialized:
            return
        
        self.ksp_is_initialized = True

        # Get PETSc Krylov solver object and set the outer KSP function; 
        # Set the root node's KSP to the outer KSP
        ksp = self.krylov_solver
        self._outer_ksp_set_func(ksp)
        self._block_split_tree.root.set_ksp(ksp)

        # Assemble matrix to define matrix size for PETSc KSP fieldsplit
        jac_form = dolfinx.fem.form(self.problem.jacobian)
        A = dolfinx.fem.petsc.assemble_matrix(jac_form, bcs=self.problem.physics.dirichlet_bcs)
        A.assemble()
        ksp.setOperators(A)

        # Preallocate b with the right layout
        res_form = dolfinx.fem.form(self.problem.weak_form)
        self._b = dolfinx.fem.petsc.create_vector(res_form)

        # For each split, built the block split preconditioner
        for split in self._block_split_tree.splits:
            blocks = split.pop('fields')
            # Initialize sub KSPs for the blocks with default set functions (override at the end)
            ksp0_set_function = split.pop('ksp0_set_function', self.default_set_ksp0)
            ksp1_set_function = split.pop('ksp1_set_function', self.default_set_ksp1)

            # Get the Index Set (IS) data (block_name, IS) for each block;
            # Assemble the sub KSPs for each block;
            # Pass block fields, IS data, and ksp objects to the child nodes
            is0_data, is1_data = self._block_split_tree.split_IS(blocks[0], blocks[1])
            ksp0, ksp1 = self._block_split_tree.build_block_split_pc(blocks[0], blocks[1], is0_data, is1_data, **split)
            self._block_split_tree.set_child_node_ksps(blocks[0], blocks[1], is0_data, is1_data, ksp0, ksp1)

            # Override the default KSP setup functions for the child nodes
            ksp0_set_function(ksp0)
            ksp1_set_function(ksp1)

    def default_set_ksp0(self, ksp):
        """
        Default KSP setup function for the root block in the block split tree.
        This is a placeholder and can be customized as needed.
        """
        super().default_set_ksp(ksp)
        ksp.setMonitor(ConvergenceMonitor('ksp0'))

    def default_set_ksp1(self, ksp):
        """
        Default KSP setup function for the first block in the block split tree.
        This is a placeholder and can be customized as needed.
        """
        super().default_set_ksp(ksp)
        ksp.setMonitor(ConvergenceMonitor('ksp1'))
            
