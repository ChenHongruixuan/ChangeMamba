import sys
from datetime import datetime

import numpy as np


def is_interactive_stream(stream=None):
    stream = stream or sys.stderr
    return hasattr(stream, "isatty") and stream.isatty()


def _format_scalar(value, precision=4):
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{precision}f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def format_value(value, precision=4):
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=precision, separator=", ", max_line_width=100000)
    if isinstance(value, (list, tuple)):
        array = np.asarray(value)
        if array.ndim > 0 and np.issubdtype(array.dtype, np.number):
            return np.array2string(array, precision=precision, separator=", ", max_line_width=100000)
        return "[" + ", ".join(_format_scalar(item, precision=precision) for item in value) + "]"
    return _format_scalar(value, precision=precision)


def format_log_block(title, values=None, meta=None, precision=4):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"[{timestamp}] {title}"
    if meta:
        meta_str = " | ".join(f"{key}={format_value(value, precision=precision)}" for key, value in meta.items())
        header = f"{header} | {meta_str}"
    if not values:
        return header

    key_width = max(len(str(key)) for key in values)
    lines = [header]
    for key, value in values.items():
        lines.append(f"  {str(key):<{key_width}} : {format_value(value, precision=precision)}")
    return "\n".join(lines)
