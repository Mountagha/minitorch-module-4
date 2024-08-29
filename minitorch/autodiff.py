from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    l = list(vals)
    l[arg] += epsilon
    f_x_plus_h = f(*l)
    l[arg] -= 2 * epsilon
    f_x_moins_h = f(*l)
    return (f_x_plus_h - f_x_moins_h) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> List[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    variables: List[Variable] = []
    visited = set()

    def _build_topological(v: Variable) -> None:
        if v.unique_id not in visited:
            visited.add(v.unique_id)
            if not v.is_constant():
                for e in v.parents:
                    _build_topological(e)
                variables.append(v)

    _build_topological(variable)
    variables.reverse()
    return variables


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    variables = topological_sort(variable)

    scalar_deriv: Dict[int, Any] = dict()
    scalar_deriv[variable.unique_id] = deriv
    for current_var in variables:
        current_deriv = scalar_deriv[current_var.unique_id]
        if current_var.is_leaf():
            current_var.accumulate_derivative(current_deriv)
        else:
            for v, d in current_var.chain_rule(current_deriv):
                if v.unique_id in scalar_deriv:
                    scalar_deriv[v.unique_id] += d
                else:
                    scalar_deriv[v.unique_id] = d


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
