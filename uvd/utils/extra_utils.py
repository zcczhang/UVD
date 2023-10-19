__all__ = ["assert_", "prepare_locals_for_super", "json_str"]


def assert_(*values, info: str = ""):
    if not len(set(values)) == 1:
        txt = f"{' != '.join(f'{v=}' for v in values)}"
        if info is not None and len(info) > 0:
            txt = info + " " + txt
        raise AssertionError(txt)


def prepare_locals_for_super(
    local_vars, args_name="args", kwargs_name="kwargs", ignore_kwargs=False
):
    assert (
        args_name not in local_vars
    ), "`prepare_locals_for_super` does not support {}.".format(args_name)
    new_locals = {k: v for k, v in local_vars.items() if k != "self" and "__" not in k}
    if kwargs_name in new_locals:
        if ignore_kwargs:
            new_locals.pop(kwargs_name)
        else:
            kwargs = new_locals.pop(kwargs_name)
            kwargs.update(new_locals)
            new_locals = kwargs
    return new_locals


def json_str(data: dict, indent: int = 4) -> str:
    def _serialize(item, level=0):
        if isinstance(item, dict):
            return "\n" + "\n".join(
                [
                    f'{" " * (level + 1) * indent}{k}: {_serialize(v, level + 1)}'
                    for k, v in item.items()
                ]
            )
        elif hasattr(item, "to_dict"):
            return item.to_dict()
        elif hasattr(item, "__class__"):
            if hasattr(item, "__repr__"):
                return item.__repr__()
            else:
                return item.__class__.__name__
        else:
            return item

    return "\n" + "\n".join(
        [f'{" " * indent}{k}: {_serialize(v)}' for k, v in data.items()]
    )
