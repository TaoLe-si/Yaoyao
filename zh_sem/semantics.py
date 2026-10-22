# -*- coding: utf-8 -*-
"""句义帧：构造、规范化、对拍。v0.5 支持嵌套补语（comps）。"""


def event(pred, roles=None, comps=None, construction="event", **mods):
    d = {
        "kind": "event",
        "pred": pred,
        "roles": dict(roles or {}),
        "mods": {k: v for k, v in mods.items() if v},
        "construction": construction,
    }
    if comps:
        d["comps"] = dict(comps)
    return d


def state(pred, roles=None, comps=None, construction="state", **mods):
    d = {
        "kind": "state",
        "pred": pred,
        "roles": dict(roles or {}),
        "mods": {k: v for k, v in mods.items() if v},
        "construction": construction,
    }
    if comps:
        d["comps"] = dict(comps)
    return d


def entity(head, base=None, comps=None, construction="entity", **mods):
    d = {
        "kind": "entity",
        "pred": head,
        "base": base or head,
        "roles": {},
        "mods": {k: v for k, v in mods.items() if v},
        "construction": construction,
    }
    if comps:
        d["comps"] = dict(comps)
    return d


def _freeze(x):
    """把任意嵌套 dict/list 变成可比较、可排序的元组。"""
    if isinstance(x, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in x.items()))
    if isinstance(x, (list, tuple)):
        return tuple(_freeze(v) for v in x)
    return x


def _role_canon(v):
    return canon(v) if isinstance(v, dict) else v


def canon(m):
    """忽略 construction 的规范形式；None 表示 REJECT。嵌套 comps/角色递归规范化。"""
    if m is None:
        return None
    comps = tuple(sorted((k, canon(v)) for k, v in m.get("comps", {}).items()))
    return (
        m["kind"],
        m["pred"],
        tuple(sorted((k, _role_canon(v)) for k, v in m["roles"].items())),
        tuple(sorted((k, v) for k, v in m["mods"].items() if v)),
        comps,
    )


def describe(m):
    if m is None:
        return "REJECT"
    roles = " ".join("%s=%s" % (k, describe(v) if isinstance(v, dict) else v)
                     for k, v in sorted(m["roles"].items()))
    mods = " ".join("%s" % k for k, v in sorted(m["mods"].items()) if v)
    s = "%s(%s) %s" % (m["pred"], roles, ("[" + mods + "]") if mods else "")
    if m.get("comps"):
        inner = " | ".join("%s=%s" % (k, describe(v))
                           for k, v in sorted(m["comps"].items()))
        s += " {%s}" % inner
    return s
