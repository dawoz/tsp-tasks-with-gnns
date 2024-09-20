from cplex.callbacks import LazyConstraintCallback, NodeCallback, BranchCallback
from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin, print_called


class LazyCallback(ConstraintCallbackMixin, LazyConstraintCallback):
    def __init__(self, env):
        LazyConstraintCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        self.nb_lazy_cts = 0  # Initialize the number of lazy constraints added to 0

    def add_lazy_constraints(self, cts):
        self.register_constraints(cts)

    @print_called('--> lazy constraint callback called: #{0}')
    def __call__(self):
        sol = self.make_solution()  # Obtain the current solution
        unsats = self.get_cpx_unsatisfied_cts(self.cts, sol, tolerance=1e-6)  # Get the unsatisfied constraints
        for ct, cpx_lhs, sense, cpx_rhs in unsats:
            self.add(cpx_lhs, sense, cpx_rhs)  # Add the unsatisfied constraint to the model
            self.nb_lazy_cts += 1  # Increment the number of lazy constraints added
            print('  -- new lazy constraint[{0}]: {1!s}'.format(self.nb_lazy_cts, ct))
            

class TorchModuleWrapper:
    def __init__(self):
        self.module = None

    def set_module(self, module, verbose=False):
        assert self.module is None, 'Module already set'
        if verbose:
            print(f'{self}: {module}')
        self.module = module


class NodeSelection_MLBased(ConstraintCallbackMixin, NodeCallback, TorchModuleWrapper):
    def __init__(self, env):
        NodeCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        TorchModuleWrapper.__init__(self)

    # DOCUMENTATION: https://www.ibm.com/docs/en/icos/22.1.1?topic=SSSA5P_22.1.1/ilog.odms.cplex.help/refpythoncplex/html/cplex.callbacks.NodeCallback-class.htm

    # @print_called('--> node callback called: #{0}')
    def __call__(self):
        # for i in range(self.get_num_nodes()):
        #     print(self.get_objective_value(i), self.get_estimated_objective_value(i))
        # print('_________________________________________________________')
        
        # print(self.model.get_var_by_index(self.get_branch_variable(0)))
        
        self.select_node(0)
    
    
class VariableSelection_MLBased(ConstraintCallbackMixin, BranchCallback, TorchModuleWrapper):
    
    # DOCUMENTATION: https://www.ibm.com/docs/en/icos/22.1.1?topic=SSSA5P_22.1.1/ilog.odms.cplex.help/refpythoncplex/html/cplex.callbacks.BranchCallback-class.htm
    
    def __init__(self, env):
        BranchCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        TorchModuleWrapper.__init__(self)

    # @print_called('--> branch callback called: #{0}')
    def __call__(self):
        for i in range(self.get_num_branches()):
            print(self.get_branch(i))
        print('_________________________________________________________')