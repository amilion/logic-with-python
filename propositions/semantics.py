# This file is part of the materials accompanying the book
# "Mathematical Logic through Python" by Gonczarowski and Nisan,
# Cambridge University Press. Book site: www.LogicThruPython.org
# (c) Yannai A. Gonczarowski and Noam Nisan, 2017-2022
# File name: propositions/semantics.py

"""Semantic analysis of propositional-logic constructs."""

from typing import AbstractSet, Iterable, Iterator, Mapping, Sequence, Tuple
from itertools import product, repeat

from propositions.syntax import *
from propositions.proofs import *

#: A model for propositional-logic formulas, a mapping from variable names to
#: truth values.
Model = Mapping[str, bool]


def is_model(model: Model) -> bool:
    """Checks if the given dictionary is a model over some set of variable
    names.

    Parameters:
        model: dictionary to check.

    Returns:
        ``True`` if the given dictionary is a model over some set of variable
        names, ``False`` otherwise.
    """
    for key in model:
        if not is_variable(key):
            return False
    return True


def variables(model: Model) -> AbstractSet[str]:
    """Finds all variable names over which the given model is defined.

    Parameters:
        model: model to check.

    Returns:
        A set of all variable names over which the given model is defined.
    """
    assert is_model(model)
    return model.keys()


def evaluate(formula: Formula, model: Model) -> bool:
    """Calculates the truth value of the given formula in the given model.

    Parameters:
        formula: formula to calculate the truth value of.
        model: model over (possibly a superset of) the variable names of the
            given formula, to calculate the truth value in.

    Returns:
        The truth value of the given formula in the given model.

    Examples:
        >>> evaluate(Formula.parse('~(p&q76)'), {'p': True, 'q76': False})
        True

        >>> evaluate(Formula.parse('~(p&q76)'), {'p': True, 'q76': True})
        False
    """
    assert is_model(model)
    assert formula.variables().issubset(variables(model))
    # Task 2.1
    if is_variable(formula.root):
        return model[formula.root]
    elif is_constant(formula.root):
        if (formula.root == 'T'):
            return True
        else:
            return False
    elif is_unary(formula.root):
        return not evaluate(formula.first, model)
    elif is_binary(formula.root):
        if formula.root == '|':
            return evaluate(formula.first, model) or evaluate(formula.second, model)
        elif formula.root == '&':
            return evaluate(formula.first, model) and evaluate(formula.second, model)
        elif formula.root == '->':
            return not evaluate(formula.first, model) or evaluate(formula.second, model)


def all_models(variables: Sequence[str]) -> Iterable[Model]:
    """Calculates all possible models over the given variable names.

    Parameters:
        variables: variable names over which to calculate the models.

    Returns:
        An iterable over all possible models over the given variable names. The
        order of the models is lexicographic according to the order of the given
        variable names, where False precedes True.

    Examples:
        >>> list(all_models(['p', 'q']))
        [{'p': False, 'q': False}, {'p': False, 'q': True}, {'p': True, 'q': False}, {'p': True, 'q': True}]

        >>> list(all_models(['q', 'p']))
        [{'q': False, 'p': False}, {'q': False, 'p': True}, {'q': True, 'p': False}, {'q': True, 'p': True}]
    """
    # Task 2.2
    for v in variables:
        assert is_variable(v)
        if not variables[1:]:
            return [{v: False}, {v: True}]
        return list(map(lambda x: x[0] | x[1], product([{v: False}, {v: True}], all_models(variables[1:]))))


def truth_values(formula: Formula, models: Iterable[Model]) -> Iterable[bool]:
    """Calculates the truth value of the given formula in each of the given
    models.

    Parameters:
        formula: formula to calculate the truth value of.
        models: iterable over models to calculate the truth value in.

    Returns:
        An iterable over the respective truth values of the given formula in
        each of the given models, in the order of the given models.

    Examples:
        >>> list(truth_values(Formula.parse('~(p&q76)'), all_models(['p', 'q76'])))
        [True, True, True, False]
    """
    # Task 2.3
    lst = []
    for model in models:
        if evaluate(formula, model):
            lst.append(True)
        else:
            lst.append(False)
    return lst


def print_truth_table(formula: Formula) -> None:
    """Prints the truth table of the given formula, with variable-name columns
    sorted alphabetically.

    Parameters:
        formula: formula to print the truth table of.

    Examples:
        >>> print_truth_table(Formula.parse('~(p&q76)'))
        | p | q76 | ~(p&q76) |
        |---|-----|----------|
        | F | F   | T        |
        | F | T   | T        |
        | T | F   | T        |
        | T | T   | F        |
    """
    # Task 2.4
    atoms = sorted(formula.variables())
    models = all_models(atoms)
    sizes = []
    headers = atoms + [formula.__repr__()]
    print("| {} |".format(' | '.join(headers)))
    for header in headers:
        sizes.append(len(header))
    for size in sizes:
        print("|-{}-".format("-" * size), end="")
    print("|")
    for ind, model in enumerate(models):
        for ind, value in enumerate(model.values()):
            print(f"| {str(value)[0]:{sizes[ind]}} ", end="")
        print(f"| {str(evaluate(formula, model))[0]:{sizes[-1]}} |")


def is_tautology(formula: Formula) -> bool:
    """Checks if the given formula is a tautology.

    Parameters:
        formula: formula to check.

    Returns:
        ``True`` if the given formula is a tautology, ``False`` otherwise.
    """
    # Task 2.5a
    models = all_models(list(formula.variables()))
    if not models:
        return evaluate(formula, {'p': True})
    for model in models:
        if not evaluate(formula, model):
            return False
    return True


def is_contradiction(formula: Formula) -> bool:
    """Checks if the given formula is a contradiction.

    Parameters:
        formula: formula to check.

    Returns:
        ``True`` if the given formula is a contradiction, ``False`` otherwise.
    """
    # Task 2.5b
    if is_tautology(Formula("~", formula)):
        return True
    return False


def is_satisfiable(formula: Formula) -> bool:
    """Checks if the given formula is satisfiable.

    Parameters:
        formula: formula to check.

    Returns:
        ``True`` if the given formula is satisfiable, ``False`` otherwise.
    """
    # Task 2.5c
    if is_contradiction(formula):
        return False
    return True


def _synthesize_for_model(model: Model) -> Formula:
    """Synthesizes a propositional formula in the form of a single conjunctive
    clause that evaluates to ``True`` in the given model, and to ``False`` in
    any other model over the same variable names.

    Parameters:
        model: model over a nonempty set of variable names, in which the
            synthesized formula is to hold.

    Returns:
        The synthesized formula.
    """
    assert is_model(model)
    assert len(model.keys()) > 0
    # Task 2.6
    keys = list(model.keys())
    clauses = []
    for key in keys:
        if model[key]:
            clauses.append(Formula(key))
        else:
            clauses.append(Formula("~", Formula(key)))
    formula = clauses[0]
    if len(clauses) > 1:
        for clause in clauses[1:]:
            formula = Formula("&", formula, clause)
    return formula


def synthesize(variables: Sequence[str], values: Iterable[bool]) -> Formula:
    """Synthesizes a propositional formula in DNF over the given variable names,
    that has the specified truth table.

    Parameters:
        variables: nonempty set of variable names for the synthesized formula.
        values: iterable over truth values for the synthesized formula in every
            possible model over the given variable names, in the order returned
            by `all_models`\ ``(``\ `~synthesize.variables`\ ``)``.

    Returns:
        The synthesized formula.

    Examples:
        >>> formula = synthesize(['p', 'q'], [True, True, True, False])
        >>> for model in all_models(['p', 'q']):
        ...     evaluate(formula, model)
        True
        True
        True
        False
    """
    assert len(variables) > 0
    # Task 2.7
    models = all_models(variables)
    ands = []
    has_true = False
    for ind, flag in enumerate(values):
        if flag:
            has_true = True
            model = models[ind]
            ands.append(_synthesize_for_model(model))
    if not has_true:
        for variable in variables:
            and_formula = Formula("&", Formula(variable), Formula("~", Formula(variable)))
            ands.append(and_formula)
    formula = ands[0]
    for clause in ands[1:]:
        formula = Formula("|", formula, clause)
    return formula


def _synthesize_for_all_except_model(model: Model) -> Formula:
    """Synthesizes a propositional formula in the form of a single disjunctive
    clause that evaluates to ``False`` in the given model, and to ``True`` in
    any other model over the same variable names.

    Parameters:
        model: model over a nonempty set of variable names, in which the
            synthesized formula is to not hold.

    Returns:
        The synthesized formula.
    """
    assert is_model(model)
    assert len(model.keys()) > 0
    # Optional Task 2.8
    literals = []
    for variable, flag in model.items():
        if flag:
            literals.append(Formula("~", Formula(variable)))
        else:
            literals.append(Formula(variable))
    formula = literals[0]
    for literal in literals[1:]:
        formula = Formula("|", formula, literal)
    return formula


def synthesize_cnf(variables: Sequence[str], values: Iterable[bool]) -> Formula:
    """Synthesizes a propositional formula in CNF over the given variable names,
    that has the specified truth table.

    Parameters:
        variables: nonempty set of variable names for the synthesized formula.
        values: iterable over truth values for the synthesized formula in every
            possible model over the given variable names, in the order returned
            by `all_models`\ ``(``\ `~synthesize.variables`\ ``)``.

    Returns:
        The synthesized formula.

    Examples:
        >>> formula = synthesize_cnf(['p', 'q'], [True, True, True, False])
        >>> for model in all_models(['p', 'q']):
        ...     evaluate(formula, model)
        True
        True
        True
        False
    """
    assert len(variables) > 0
    # Optional Task 2.9
    has_false = 0
    clauses = []
    models = all_models(variables)
    for ind, flag in enumerate(values):
        if not flag:
            has_false = 1
            clause = _synthesize_for_all_except_model(models[ind])
            clauses.append(clause)
    if not has_false:
        for variable in variables:
            clause = Formula("|", Formula(variable), Formula("~", Formula(variable)))
            clauses.append(clause)
    formula = clauses[0]
    for clause in clauses[1:]:
        formula = Formula("&", formula, clause)
    return formula


def evaluate_inference(rule: InferenceRule, model: Model) -> bool:
    """Checks if the given inference rule holds in the given model.

    Parameters:
        rule: inference rule to check.
        model: model to check in.

    Returns:
        ``True`` if the given inference rule holds in the given model, ``False``
        otherwise.

    Examples:
        >>> evaluate_inference(InferenceRule([Formula('p')], Formula('q')),
        ...                    {'p': True, 'q': False})
        False

        >>> evaluate_inference(InferenceRule([Formula('p')], Formula('q')),
        ...                    {'p': False, 'q': False})
        True
    """
    assert is_model(model)
    # Task 4.2


def is_sound_inference(rule: InferenceRule) -> bool:
    """Checks if the given inference rule is sound, i.e., whether its
    conclusion is a semantically correct implication of its assumptions.

    Parameters:
        rule: inference rule to check.

    Returns:
        ``True`` if the given inference rule is sound, ``False`` otherwise.
    """
    # Task 4.3
