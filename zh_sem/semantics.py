# -*- coding: utf-8 -*-
"""句义帧：构造、规范化、对拍。"""


def event(pred, roles=None, construction="event", **mods):
    return {
        "kind": "event",
        "pred": pred,
        "roles": dict(roles or {}),
        "mods": {k: v for k, v in mods.items() if v},
        "construction": construction,
    }


def state(pred, roles=None, construction="state", **mods):
    return {
        "kind": "state",
        "pred": pred,
        "roles": dict(roles or {}),
        "mods": {k: v for k, v in mods.items() if v},
        "construction": construction,
    }


def entity(head, base=None, construction="entity", **mods):
    return {
        "kind": "entity",
        "pred": head,
        "base": base or head,
        "roles": {},
        "mods": {k: v for k, v in mods.items() if v},
        "construction": construction,
    }


def canon(m):
    """忽略 construction 的规范形式；None 表示 REJECT。"""
    if m is None:
        return None
    return (
        m["kind"],
        m["pred"],
        tuple(sorted(m["roles"].items())),
        tuple(sorted((k, v) for k, v in m["mods"].items() if v)),
    )


def describe(m):
    if m is None:
        return "REJECT"
    roles = " ".join("%s=%s" % (k, v) for k, v in sorted(m["roles"].items()))
    mods = " ".join("%s" % k for k, v in sorted(m["mods"].items()) if v)
    return "%s(%s) %s" % (m["pred"], roles, ("[" + mods + "]") if mods else "")
