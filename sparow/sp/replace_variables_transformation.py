from pyomo.core import (
    Constraint,
    Objective,
    Var,
    BooleanVar,
    Expression,
    Suffix,
    Param,
    Set,
    SetOf,
    RangeSet,
    ExternalFunction,
    Connector,
    SortComponents,
    Any,
    LogicalConstraint,
)
from pyomo.core.base import Transformation
from pyomo.core.base.block import Block

from pyomo.gdp import Disjunct, Disjunction
#from pyomo.gdp.util import is_child_of
from pyomo.network import Port

from pyomo.core.expr import ExpressionReplacementVisitor


class ReplaceVariablesTransformation(Transformation):
    """
    Replace variables in a model to new variable data objects.
    """

    def __init__(self):
        super().__init__()
        self.handlers = {
            Constraint: self._transform_constraint,
            Objective: self._transform_objective,
            Block: self._transform_block,
            Var: False,
            BooleanVar: False,
            Connector: False,
            Expression: False,
            Suffix: False,
            Param: False,
            Set: False,
            SetOf: False,
            RangeSet: False,
            Disjunction: False,
            Disjunct: False,
            ExternalFunction: False,
            Port: False,
            LogicalConstraint: False,
        }

    def _apply_to(self, instance, substitution_map):
        try:
            self._apply_to_impl(instance, substitution_map)
        finally:
            pass

    def _apply_to_impl(self, instance, substitution_map):
        self.visitor = ExpressionReplacementVisitor(
            substitute=substitution_map,
            descend_into_named_expressions=True,
            remove_named_expressions=False,
        )

        t = instance
        if issubclass(t.ctype, Block):
            self._transform_block(t)
        elif t.ctype is Constraint:
            self._transform_constraint(t)
        elif t.ctype is Objective:
            self._transform_objective(t)
        else:
            raise ValueError(
                f"Target '{t.name}' is not a Block, Objective or "
                "Constraint. It was of type '{type(t)}' and cannot be transformed."
            )

    def _transform_block(self, block):
        blocks = block.values() if block.is_indexed() else (block,)
        for b in blocks:
            for obj in b.component_objects(
                active=True,
                descend_into=(Block, Disjunct),
                sort=SortComponents.deterministic,
            ):
                handler = self.handlers.get(obj.ctype, None)
                if not handler:
                    if handler is None:
                        raise RuntimeError(
                            f"No transformation handler registered for modeling "
                            "components of type '{obj.ctype}'."
                        )
                    continue
                handler(obj)

    def _transform_constraint(self, constraint):
        constraints = constraint.values() if constraint.is_indexed() else (constraint,)
        for c in constraints:
            orig_body = c.body
            new_body = self.visitor.walk_expression(orig_body)
            if orig_body is not new_body:
                c.set_value((c.lower, new_body, c.upper))

    def _transform_objective(self, objective):
        objectives = objective.values() if objective.is_indexed() else (objective,)
        for o in objectives:
            orig_expr = o.expr
            new_expr = self.visitor.walk_expression(orig_expr)
            if orig_expr is not new_expr:
                o.set_value(new_expr)
