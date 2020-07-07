from collections import OrderedDict
from mpi4py import MPI
import numpy
import itertools
from firedrake.ufl_expr import adjoint, action
from firedrake.formmanipulation import ExtractSubBlock
from firedrake.bcs import DirichletBC, EquationBCSplit

from firedrake.petsc import PETSc
from firedrake.utils import cached_property

from firedrake.constant import Constant

__all__ = ("ImplicitMatrixContext", )


def find_sub_block(iset, ises):
    """Determine if iset comes from a concatenation of some subset of
    ises.

    :arg iset: a PETSc IS to find in ``ises``.
    :arg ises: An iterable of PETSc ISes.

    :returns: The indices into ``ises`` that when concatenated
        together produces ``iset``.

    :raises LookupError: if ``iset`` could not be found in
        ``ises``.
    """
    found = []
    comm = iset.comm
    target_indices = iset.indices
    comm = iset.comm.tompi4py()
    candidates = OrderedDict(enumerate(ises))
    while True:
        match = False
        for i, candidate in list(candidates.items()):
            candidate_indices = candidate.indices
            candidate_size, = candidate_indices.shape
            target_size, = target_indices.shape
            # Does the local part of the candidate IS match a prefix
            # of the target indices?
            lmatch = (candidate_size <= target_size
                      and numpy.array_equal(target_indices[:candidate_size], candidate_indices))
            if comm.allreduce(lmatch, op=MPI.LAND):
                # Yes, this candidate matched, so remove it from the
                # target indices, and list of candidate
                target_indices = target_indices[candidate_size:]
                found.append(i)
                candidates.pop(i)
                # And keep looking for the remainder in the remaining candidates.
                match = True
        if not match:
            break
    if comm.allreduce(len(target_indices), op=MPI.SUM) > 0:
        # We didn't manage to hoover up all the target indices, not a match
        raise LookupError("Unable to find %s in %s" % (iset, ises))
    return found

def find_element_of_which_sub_block(rows,ises):
    # This function acts as lookup to find which block the indices belong in
    block = {}
    shift = []
    found = 0
    candidates = OrderedDict(enumerate(ises))
    for i, candidate in list(candidates.items()):
        # Initialise dictionary to hold the rows and the shift parameter
        # since DirichletBC starts from zero for each block
        block[i] = []
        shift.append(candidate.indices[0])
    for row in rows:
        for i, candidate in list(candidates.items()):
            candidate_indices = candidate.indices
            lmatch = numpy.isin(row, candidate_indices)
            # We found the block in which the index lives, so we store it
            #if comm.allreduce(lmatch, op=MPI.LAND):
            if lmatch:
                block[i].append(row)
                found += 1
                break
    if found < len(rows):
        # We did not manage to find the row in the possible index sets
        raise LookupError("Unable to find %s in %s" % (rows, ises))
    return (block, shift)




class ZeroRowsColumnsBC(DirichletBC):
  """
   This overloads the DirichletBC class in order to impose homogeneous Dirichlet boundary
   conditions on user-defined vertices
  """
  def __init__(self, V, val, rows = None, sub_domain = "on_boundary", method="topological"):
      super().__init__(V, val, [], method)
      if rows is not None:
          self.nodes = numpy.array(rows)

  def reconstruct(self, field=None, V=None, g=None, sub_domain=None, method=None, use_split=False):
      fs = self.function_space()
      if V is None:
          V = fs
      if g is None:
         g = self._original_arg
      if sub_domain is None:
         sub_domain = self.sub_domain
      if method is None:
         method = self.method
      if field is not None:
         assert V is not None, "`V` can not be `None` when `field` is not `None`"
         V = self.as_subspace(field, V, use_split)
         if V is None:
             return
      if V == fs and \
         V.parent == fs.parent and \
         V.index == fs.index and \
         (V.parent is None or V.parent.parent == fs.parent.parent) and \
         (V.parent is None or V.parent.index == fs.parent.index) and \
         g == self._original_arg and \
         sub_domain == self.sub_domain and method == self.method:
             return self
      return type(self)(V, g, rows = self.nodes, sub_domain = "on_boundary", method=method)


class ImplicitMatrixContext(object):
    # By default, these matrices will represent diagonal blocks (the
    # (0,0) block of a 1x1 block matrix is on the diagonal).
    on_diag = True

    """This class gives the Python context for a PETSc Python matrix.

    :arg a: The bilinear form defining the matrix

    :arg row_bcs: An iterable of the :class.`.DirichletBC`s that are
      imposed on the test space.  We distinguish between row and
      column boundary conditions in the case of submatrices off of the
      diagonal.

    :arg col_bcs: An iterable of the :class.`.DirichletBC`s that are
       imposed on the trial space.

    :arg fcparams: A dictionary of parameters to pass on to the form
       compiler.

    :arg appctx: Any extra user-supplied context, available to
       preconditioners and the like.

    """
    def __init__(self, a, row_bcs=[], col_bcs=[],
                 fc_params=None, appctx=None):
        self.a = a
        self.aT = adjoint(a)
        self.fc_params = fc_params
        self.appctx = appctx

        # Collect all DirichletBC instances including
        # DirichletBCs applied to an EquationBC.

        # all bcs (DirichletBC, EquationBCSplit)
        self.bcs = row_bcs
        self.bcs_col = col_bcs
        self.row_bcs = tuple(bc for bc in itertools.chain(*row_bcs) if isinstance(bc, DirichletBC))
        self.col_bcs = tuple(bc for bc in itertools.chain(*col_bcs) if isinstance(bc, DirichletBC))

        # create functions from test and trial space to help
        # with 1-form assembly
        test_space, trial_space = [
            a.arguments()[i].function_space() for i in (0, 1)
        ]
        from firedrake import function
        self._y = function.Function(test_space)
        self._x = function.Function(trial_space)
        
        # Temporary storage for holding the BC values during zeroRowsColumns
        self._tmp_zeroRowsColumns = function.Function(test_space)
        # These are temporary storage for holding the BC
        # values during matvec application.  _xbc is for
        # the action and ._ybc is for transpose.
        if len(self.row_bcs) > 0:
            self._xbc = function.Function(trial_space)
        if len(self.col_bcs) > 0:
            self._ybc = function.Function(test_space)

        # Get size information from template vecs on test and trial spaces
        trial_vec = trial_space.dof_dset.layout_vec
        test_vec = test_space.dof_dset.layout_vec
        self.col_sizes = trial_vec.getSizes()
        self.row_sizes = test_vec.getSizes()

        self.block_size = (test_vec.getBlockSize(), trial_vec.getBlockSize())

        self.action = action(self.a, self._x)
        self.actionT = action(self.aT, self._y)

        from firedrake.assemble import create_assembly_callable

        # For assembling action(f, self._x)
        self.bcs_action = []
        for bc in self.bcs:
            if isinstance(bc, DirichletBC):
                self.bcs_action.append(bc)
            elif isinstance(bc, EquationBCSplit):
                self.bcs_action.append(bc.reconstruct(action_x=self._x))

        self._assemble_action = create_assembly_callable(self.action, tensor=self._y, bcs=self.bcs_action,
                                                         form_compiler_parameters=self.fc_params)

        # For assembling action(adjoint(f), self._y)
        # Sorted list of equation bcs
        self.objs_actionT = []
        for bc in self.bcs:
            self.objs_actionT += bc.sorted_equation_bcs()
        self.objs_actionT.append(self)
        # Each par_loop is to run with appropriate masks on self._y
        self._assemble_actionT = []
        # Deepest EquationBCs first
        for bc in self.bcs:
            for ebc in bc.sorted_equation_bcs():
                self._assemble_actionT.append(create_assembly_callable(action(adjoint(ebc.f), self._y),
                                              tensor=self._x,
                                              bcs=None,
                                              form_compiler_parameters=self.fc_params))
        # Domain last
        self._assemble_actionT.append(create_assembly_callable(self.actionT,
                                                               tensor=self._x,
                                                               bcs=None,
                                                               form_compiler_parameters=self.fc_params))

    @cached_property
    def _diagonal(self):
        from firedrake import Function
        assert self.on_diag
        return Function(self._x.function_space())

    @cached_property
    def _assemble_diagonal(self):
        from firedrake.assemble import create_assembly_callable
        return create_assembly_callable(self.a,
                                        tensor=self._diagonal,
                                        form_compiler_parameters=self.fc_params,
                                        diagonal=True)

    def getDiagonal(self, mat, vec):
        self._assemble_diagonal()
        for bc in self.bcs:
            # Operator is identity on boundary nodes
            bc.set(self._diagonal, 1)
        with self._diagonal.dat.vec_ro as v:
            v.copy(vec)

    def missingDiagonal(self, mat):
        return (False, -1)

    def mult(self, mat, X, Y):
        with self._x.dat.vec_wo as v:
            X.copy(v)

        # if we are a block on the diagonal, then the matrix has an
        # identity block corresponding to the Dirichlet boundary conditions.
        # our algorithm in this case is to save the BC values, zero them
        # out before computing the action so that they don't pollute
        # anything, and then set the values into the result.
        # This has the effect of applying
        # [ A_II 0 ; 0 I ] where A_II is the block corresponding only to
        # non-fixed dofs and I is the identity block on the fixed dofs.

        # If we are not, then the matrix just has 0s in the rows and columns.
        for bc in self.col_bcs:
            bc.zero(self._x)
        self._assemble_action()
        # This sets the essential boundary condition values on the
        # result.
        if self.on_diag:
            if len(self.row_bcs) > 0:
                # TODO, can we avoid the copy?
                with self._xbc.dat.vec_wo as v:
                    X.copy(v)
            for bc in self.row_bcs:
                bc.set(self._y, self._xbc)
        else:
            for bc in self.row_bcs:
                bc.zero(self._y)

        with self._y.dat.vec_ro as v:
            v.copy(Y)

    def multTranspose(self, mat, Y, X):
        """
        EquationBC makes multTranspose different from mult.

        Decompose M^T into bundles of columns associated with
        the rows of M corresponding to cell, facet,
        edge, and vertice equations (if exist) and add up their
        contributions.

                           Domain
            a a a a 0 a a    |
            a a a a 0 a a    |
            a a a a 0 a a    |   EBC1
        M = b b b b b b b    |    |   EBC2 DBC1
            0 0 0 0 1 0 0    |    |    |    |
            c c c c 0 c c    |         |
            c c c c 0 c c    |         |
                                                     To avoid copys, use same _y, and update it
                                                     from left (deepest ebc) to right (least deep ebc or domain)
        Multiplication algorithm:                       _y         update ->     _y        update ->   _y

                 a a a b 0 c c   _y0     0 0 0 0 c c c   *      0 0 0 b b 0 0    *     a a a a a a a   _y0          0
                 a a a b 0 c c   _y1     0 0 0 0 c c c   *      0 0 0 b b 0 0    *     a a a a a a a   _y1          0
                 a a a b 0 c c   _y2     0 0 0 0 c c c   *      0 0 0 b b 0 0    *     a a a a a a a   _y2          0
        M^T _y = a a a b 0 c c   _y3  =  0 0 0 0 c c c   *    + 0 0 0 b b 0 0   _y3  + a a a a a a a    0      +    0
                 0 0 0 0 1 0 0   _y4     0 0 0 0 c c c   0      0 0 0 b b 0 0    0     a a a a a a a    0          _y4 (replace at the end)
                 a a a b 0 c c   _y5     0 0 0 0 c c c   _y5    0 0 0 b b 0 0    *     a a a a a a a    0           0
                 a a a b 0 c c   _y6     0 0 0 0 c c c   _y6    0 0 0 b b 0 0    *     a a a a a a a    0           0
                                             (uniform on           (uniform          (uniform on domain)
                                              on facet2)            on facet1)

        * = can be any number

        """
        # accumulate values in self._xbc for convenience
        self._xbc.dat.zero()
        with self._y.dat.vec_wo as v:
            Y.copy(v)

        # Apply actionTs in sorted order
        for aT, obj in zip(self._assemble_actionT, self.objs_actionT):
            # zero columns associated with DirichletBCs/EquationBCs
            for obc in obj.bcs:
                obc.zero(self._y)
            aT()
            self._xbc += self._x

        if self.on_diag:
            if len(self.col_bcs) > 0:
                # TODO, can we avoid the copy?
                with self._ybc.dat.vec_wo as v:
                    Y.copy(v)
                for bc in self.col_bcs:
                    bc.set(self._xbc, self._ybc)
        else:
            for bc in self.col_bcs:
                bc.zero(self._xbc)

        with self._xbc.dat.vec_ro as v:
            v.copy(X)

    def view(self, mat, viewer=None):
        if viewer is None:
            return
        typ = viewer.getType()
        if typ != PETSc.Viewer.Type.ASCII:
            return
        viewer.printfASCII("Firedrake matrix-free operator %s\n" %
                           type(self).__name__)

    def getInfo(self, mat, info=None):
        from mpi4py import MPI
        memory = self._x.dat.nbytes + self._y.dat.nbytes
        if hasattr(self, "_xbc"):
            memory += self._xbc.dat.nbytes
        if hasattr(self, "_ybc"):
            memory += self._ybc.dat.nbytes
        if info is None:
            info = PETSc.Mat.InfoType.GLOBAL_SUM
        if info == PETSc.Mat.InfoType.LOCAL:
            return {"memory": memory}
        elif info == PETSc.Mat.InfoType.GLOBAL_SUM:
            gmem = mat.comm.tompi4py().allreduce(memory, op=MPI.SUM)
            return {"memory": gmem}
        elif info == PETSc.Mat.InfoType.GLOBAL_MAX:
            gmem = mat.comm.tompi4py().allreduce(memory, op=MPI.MAX)
            return {"memory": gmem}
        else:
            raise ValueError("Unknown info type %s" % info)

    # Now, to enable fieldsplit preconditioners, we need to enable submatrix
    # extraction for our custom matrix type.  Note that we are splitting UFL
    # and index sets rather than an assembled matrix, keeping matrix
    # assembly deferred as long as possible.
    def createSubMatrix(self, mat, row_is, col_is, target=None):
        if target is not None:
            # Repeat call, just return the matrix, since we don't
            # actually assemble in here.
            target.assemble()
            return target

        # These are the sets of ISes of which the the row and column
        # space consist.
        row_ises = self._y.function_space().dof_dset.field_ises
        col_ises = self._x.function_space().dof_dset.field_ises

        row_inds = find_sub_block(row_is, row_ises)
        if row_is == col_is and row_ises == col_ises:
            col_inds = row_inds
        else:
            col_inds = find_sub_block(col_is, col_ises)

        splitter = ExtractSubBlock()
        asub = splitter.split(self.a,
                              argument_indices=(row_inds, col_inds))
        Wrow = asub.arguments()[0].function_space()
        Wcol = asub.arguments()[1].function_space()

        row_bcs = []
        col_bcs = []

        for bc in self.bcs:
            if isinstance(bc, DirichletBC):
                bc_temp = bc.reconstruct(field=row_inds, V=Wrow, g=bc.function_arg, sub_domain=bc.sub_domain, method=bc.method, use_split=True)
            elif isinstance(bc, EquationBCSplit):
                bc_temp = bc.reconstruct(field=row_inds, V=Wrow, row_field=row_inds, col_field=col_inds, use_split=True)
            if bc_temp is not None:
                row_bcs.append(bc_temp)

        if Wrow == Wcol and row_inds == col_inds and self.bcs == self.bcs_col:
            col_bcs = row_bcs
        else:
            for bc in self.bcs_col:
                if isinstance(bc, DirichletBC):
                    bc_temp = bc.reconstruct(field=col_inds, V=Wcol, g=bc.function_arg, sub_domain=bc.sub_domain, method=bc.method, use_split=True)
                elif isinstance(bc, EquationBCSplit):
                    bc_temp = bc.reconstruct(field=col_inds, V=Wcol, row_field=row_inds, col_field=col_inds, use_split=True)
                if bc_temp is not None:
                    col_bcs.append(bc_temp)

        submat_ctx = ImplicitMatrixContext(asub,
                                           row_bcs=row_bcs,
                                           col_bcs=col_bcs,
                                           fc_params=self.fc_params,
                                           appctx=self.appctx)
        submat_ctx.on_diag = self.on_diag and row_inds == col_inds
        submat = PETSc.Mat().create(comm=mat.comm)
        submat.setType("python")
        submat.setSizes((submat_ctx.row_sizes, submat_ctx.col_sizes),
                        bsize=submat_ctx.block_size)
        submat.setPythonContext(submat_ctx)
        submat.setUp()

        return submat
    
    def duplicate(self, mat, newmat):
        import ipdb; ipdb.set_trace()
        newmat_ctx = ImplicitMatrixContext(self.a,
                                           row_bcs=self.bcs,
                                           col_bcs=self.bcs_col,
                                           fc_params=self.fc_params,
                                           appctx=self.appctx)
        newmat = PETSc.Mat().create(comm=mat.comm)
        newmat.setType("python")
        newmat.setSizes((newmat_ctx.row_sizes, newmat_ctx.col_sizes),
                        bsize=newmat_ctx.block_size)
        newmat.setPythonContext(newmat_ctx)
        newmat.setUp()
        return newmat

    def zeroRowsColumns(self, mat, active_rows, diag=1.0, x=None, b=None):
        """
        The way we zero rows and columns of unassembled matrices is by
        constructing a DirichetBC corresponding to the rows and columns
        which by nature of how bcs are implemented, is equivalent to zeroing
        the rows and columns and adding a 1 to the diagonal

        These are the sets of ISes of which the the row and column
        space consist.
        """
        print("inside zerorowscolumns")
        print(b) 
        print(x)
        if active_rows is None:
            raise NotImplementedError("Matrix-free zeroRowsColumns called but no rows provided")
        if not numpy.allclose(diag, 1.0):
            raise NotImplementedError("We do not know how to implement matrix-free ZeroRowsColumns with diag not equal to 1")
        if b is None:
            print("WARNING: No right-hand side vector provided to matrix-free zeroRowsColumns, ksp.solve() may cause unexpected behaviour")
        
        ises = self._y.function_space().dof_dset.field_ises

        # Find the blocks which the rows are a part of and find the row shift
        # since DirichletBC starts from 0 for each block
        (block, shift) = find_element_of_which_sub_block(active_rows, ises)

        # Include current DirichletBC conditions
        bcs = []
        bcs_col = []
        Vrow = self._y.function_space()
        Vcol = self._x.function_space()
        [bcs.append(bc) for bc in self.bcs]
        [bcs_col.append(bc) for bc in self.bcs_col]

        # If rows and columns bcs are equal, then no need to redo columns bcs
        bcs_row_and_column_equal = self.bcs == self.bcs_col
        
        # If optional vector of solutions for zeroed rows given then need to pass 
        # to DirichletBC otherwise it will be zero
        if x:
            self._tmp_zeroRowsColumns.vector().set_local(x)
        else:
            self._tmp_zeroRowsColumns.vector().set_local(0)

        for i in range(len(block)):
            # For each block create a new DirichletBC corresponding to the
            # rows and columns to be zeroed
            if block[i]:
                rows = block[i] 
                rows = rows - shift[i]
                tmp_sub = self._tmp_zeroRowsColumns.split()[i]
                                     
                activebcs_row = ZeroRowsColumnsBC(Vrow.sub(i), tmp_sub, rows = rows)
                bcs.append(activebcs_row)
                if bcs_row_and_column_equal:
                   bcs_col.append(activebcs_row)
                else:
                    activebcs_col = ZeroRowsColumnsBC(Vcol.sub(i), tmp_sub, rows = rows)
                    bcs_col.append(activebcs_col)

        # Update bcs list
        self.bcs = tuple(bcs)
        self.bcs_col = tuple(bcs_col)
        # Set new context, so PETSc mat is aware of new bcs
        newmat_ctx = ImplicitMatrixContext(self.a,
                                           row_bcs=self.bcs,
                                           col_bcs=self.bcs_col,
                                           fc_params=self.fc_params,
                                           appctx=self.appctx)

        mat.setPythonContext(newmat_ctx)
        # Needed for MG purposes! This lets the DM SNES context aware of the new Dirichlet BCS
        # which is where the bcs are extracted from when coarsening.
        if self._x.function_space().dm.appctx:
            self._x.function_space().dm.appctx[0]._problem.bcs = tuple(bcs)
        # adjust active-set rows in residual
        if x and b:
            b.array[rows] = x.array_r[rows]
