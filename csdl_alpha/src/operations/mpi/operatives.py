from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation, ComposedOperation
from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize, validate_and_variablize
import csdl_alpha.utils.testing_utils as csdl_tests
from csdl_alpha.utils.typing import VariableLike
from typing import Literal
from mpi4py import MPI

from csdl_alpha.src.operations.custom.custom import CustomExplicitOperation
import numpy as np

class CustomAllReduce(CustomExplicitOperation):
    def __init__(self, vjp_mode:Literal["passthrough", "allreduce"], comm):
        super().__init__()
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.vjp_mode = vjp_mode

    def evaluate(self, var):
        self.declare_input('in_var', var)
        f_global = self.create_output('out_var',  var.shape)
        return f_global

    def compute(self, inputs, outputs):
        from mpi4py import MPI
        outputs['out_var'] = self.comm.allreduce(inputs['in_var'], op=MPI.SUM)

    def compute_jacvec_product(self, input_vals, outputs_vals, d_inputs, d_outputs, mode):
        from mpi4py import MPI
        if mode == 'fwd':
            # y = allreduce(x), so dy = allreduce(dx)
            if 'in_var' in d_inputs and d_inputs['in_var'] is not None:
                d_outputs['out_var'] += self.comm.allreduce(d_inputs['in_var'], op=MPI.SUM)

        elif mode == 'rev':
            if self.vjp_mode == "passthrough":
                # y = sum_r x_r, so (∂y/∂x_r)^T λ_y = λ_y on each rank
                if 'out_var' in d_outputs and d_outputs['out_var'] is not None:
                    d_inputs['in_var'] += d_outputs['out_var']

            elif self.vjp_mode == "allreduce":
                d_inputs['in_var'] += self.comm.allreduce(d_outputs['out_var'], op=MPI.SUM)

        else:
            raise ValueError(f"Unsupported mode '{mode}'. Use 'fwd' or 'rev'.")
    
def mpi_sum(var, comm):
    sum = CustomAllReduce(vjp_mode="passthrough", comm=comm).evaluate(var)
    return sum

def mpi_allreduce(var, comm):
    sum = CustomAllReduce(vjp_mode="allreduce", comm=comm).evaluate(var)
    return sum

class CustomMPIIndex(CustomExplicitOperation):
    def __init__(self, i0:int, i1:int, comm):
        super().__init__()
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.i0 = i0
        self.i1 = i1
        self.n = i1 - i0
        assert self.n > 0, f"Scatter range must be positive, got {self.n} for range ({i0}, {i1})"

    def evaluate(self, var):
        self.declare_input('in_var', var)
        self.new_shape = (self.n,) + var.shape[1:]
        f_global = self.create_output('out_var',  self.new_shape)
        return f_global

    def compute(self, inputs, outputs):
        from mpi4py import MPI
        outputs['out_var'] = inputs['in_var'][self.i0:self.i1]

    def compute_jacvec_product(self, input_vals, outputs_vals, d_inputs, d_outputs, mode):
        from mpi4py import MPI
        if mode == 'rev':
            # d_inputs == allgather
            temp_d_inputs = np.zeros_like(input_vals['in_var'])
            temp_d_inputs[self.i0:self.i1] = d_outputs['out_var']
            d_in = (self.comm.allgather(temp_d_inputs))
            d_inputs['in_var'] = np.sum(d_in, axis=0)

def index_splitter(i0:int,i1:int,comm):
    return lambda x: CustomMPIIndex(i0, i1, comm).evaluate(x)