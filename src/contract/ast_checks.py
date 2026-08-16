from __future__ import annotations

import ast

from .spec import StrategyError

FILL_FUNCS = {"zeros", "ones", "full", "zeros_like", "ones_like", "full_like"}


def _find_def(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def check_definitions_and_signature(source: str) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise StrategyError(
            f"validation 1: candidate is not valid Python ({exc.__class__.__name__}: {exc}). "
            "Return only the strategy module as a single ```python ... ``` code block."
        ) from exc
    get_strategy = _find_def(tree, "get_strategy")
    simulate = _find_def(tree, "simulate")
    if get_strategy is None:
        raise StrategyError("validation 1: missing function `get_strategy` (returning the strategy dict).")
    if simulate is None:
        raise StrategyError("validation 1: missing function `simulate`.")
    params = [a.arg for a in simulate.args.args]
    if simulate.args.vararg is not None:
        raise StrategyError(f"validation 1: `simulate` must not use *args (found *{simulate.args.vararg.arg}).")
    if simulate.args.kwarg is not None:
        raise StrategyError(f"validation 1: `simulate` must not use **kwargs (found **{simulate.args.kwarg.arg}).")
    expected = ("close", "high", "low", "volume", "macro", "x")
    if params != list(expected):
        raise StrategyError(
            f"validation 1: `simulate` signature must be exactly `simulate({', '.join(expected)})`, "
            f"got `simulate({', '.join(params)})`."
        )


def _extract_literal_list(node) -> list:
    if node is None:
        return None
    try:
        val = ast.literal_eval(node)
    except Exception:
        return None
    if isinstance(val, (list, tuple)):
        return list(val)
    return None


def check_declared_params(source: str) -> dict:
    """Check 2: extract the declared {name, variables, bounds} literal from get_strategy."""
    tree = ast.parse(source)
    fn = _find_def(tree, "get_strategy")
    if fn is None:
        raise StrategyError("validation 2: `get_strategy` missing.")
    ret = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            ret = node.value
            break
    if ret is None:
        raise StrategyError(
            "validation 2: `get_strategy` must `return {{'name': ..., 'variables': [...], "
            "'bounds': ([...], [...])}}` with literal lists."
        )
    declared = {}
    for key_node, val_node in zip(ret.keys, ret.values):
        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
            declared[key_node.value] = val_node

    name_node = declared.get("name")
    if not isinstance(name_node, ast.Constant) or not isinstance(name_node.value, str):
        raise StrategyError("validation 2: `name` must be a string literal.")
    variables = _extract_literal_list(declared.get("variables"))
    if variables is None:
        raise StrategyError("validation 2: `variables` must be a literal list of strings.")
    if len(variables) != len(set(variables)):
        raise StrategyError("validation 2: `variables` contains duplicates.")
    for v in variables:
        if not isinstance(v, str) or not v.isidentifier():
            raise StrategyError(
                f"validation 2: variable `{v!r}` is not an identifier-like string name."
            )
    from ..config.settings import MAX_VARIABLES, MIN_VARIABLES

    if not (MIN_VARIABLES <= len(variables) <= MAX_VARIABLES):
        raise StrategyError(
            f"validation 2: expected {MIN_VARIABLES}..{MAX_VARIABLES} variables, "
            f"got {len(variables)}."
        )
    bounds = declared.get("bounds")
    if not isinstance(bounds, (ast.Tuple, ast.List)) or len(bounds.elts) != 2:
        raise StrategyError(
            "validation 2: `bounds` must be a 2-tuple `(lower, upper)` where each element is a list."
        )
    lower = _extract_literal_list(bounds.elts[0])
    upper = _extract_literal_list(bounds.elts[1])
    if lower is None or upper is None:
        raise StrategyError("validation 2: `bounds` elements must be literal lists of numbers.")
    if len(lower) != len(variables) or len(upper) != len(variables):
        raise StrategyError(
            "validation 2: `bounds` lower/upper lists must have the same length as `variables` "
            f"({len(variables)}), got {len(lower)}/{len(upper)}."
        )
    if any(not isinstance(a, (int, float)) for a in lower + upper):
        raise StrategyError("validation 2: `bounds` values must be numbers.")
    for i, (lo, hi) in enumerate(zip(lower, upper)):
        if lo >= hi:
            raise StrategyError(
                f"validation 2: variable {variables[i]!r} has lower >= upper ({lo} >= {hi})."
            )
    return {"name": name_node.value, "variables": variables, "lower": lower, "upper": upper}


def _no_bare_except(node: ast.AST) -> list[str]:
    problems = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.ExceptHandler):
            if sub.type is None:
                problems.append("bare `except:`")
            elif isinstance(sub.type, ast.Name) and sub.type.id == "Exception":
                problems.append("`except Exception:`")
            elif isinstance(sub.type, ast.Name) and sub.type.id == "ZeroDivisionError":
                problems.append("`except ZeroDivisionError:`")
    return problems


def check_no_broad_except(source: str) -> None:
    tree = ast.parse(source)
    problems = _no_bare_except(tree)
    if problems:
        raise StrategyError(
            "validation 3: banned exception handling pattern found: "
            + ", ".join(sorted(set(problems)))
            + ". This masks real bugs; let errors surface."
        )


def _return_value_is_constant_fill(ret_value: ast.AST) -> bool:
    for node in ast.walk(ret_value):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr in FILL_FUNCS:
                return True
            if isinstance(fn, ast.Name) and fn.id in FILL_FUNCS:
                return True
    if isinstance(ret_value, (ast.List, ast.Tuple)):
        if all(isinstance(e, ast.Constant) for e in ret_value.elts):
            return True
    return False


def check_no_constant_fallback(source: str) -> None:
    tree = ast.parse(source)
    simulate = _find_def(tree, "simulate")
    if simulate is None:
        return
    for sub in ast.walk(simulate):
        if isinstance(sub, ast.Return) and sub.value is not None:
            if _return_value_is_constant_fill(sub.value):
                raise StrategyError(
                    "validation 4: `simulate` appears to return a hardcoded constant "
                    "positions array (all-zeros/all-ones regardless of input). "
                    "A strategy must not cheat a crash into a disguised flat result."
                )


def validate_static(source: str) -> dict:
    check_definitions_and_signature(source)
    declared = check_declared_params(source)
    check_no_broad_except(source)
    check_no_constant_fallback(source)
    return declared
