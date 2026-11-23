"""Offline scoring of the prospective seven-case set. Takes decoded continuations only."""
import json
from decimal import Decimal
from pathlib import Path


def strict_json(text):
    def unique(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError('duplicate field')
            obj[key] = value
        return obj
    def reject_constant(value):
        raise ValueError('nonfinite number')
    return json.loads(text, parse_float=Decimal, parse_int=Decimal,
                      parse_constant=reject_constant, object_pairs_hook=unique)


def same(a, b):
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x,y) for x,y in zip(a,b))
    return a == b


def score(outputs):
    spec = strict_json(Path(__file__).with_name('seven-prospective.json').read_text())
    if type(outputs) is not dict or set(outputs) != {case['id'] for case in spec}:
        raise ValueError('exactly one continuation is required per frozen case')
    result = {}
    for case in spec:
        key = case['id']
        value = outputs[key]
        if type(value) is not str:
            raise ValueError('continuation must be text')
        criteria = case['criteria']
        if 'json' in criteria:
            try:
                result[key] = same(strict_json(value), criteria['json'])
            except (ValueError, RecursionError):
                result[key] = False
        elif 'exact' in criteria:
            result[key] = value.strip() == criteria['exact']
        else:
            result[key] = value.strip().casefold() == criteria['exact_casefold']
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('continuations', type=Path)
    results = score(strict_json(parser.parse_args().continuations.read_text()))
    print(json.dumps(dict(prospective_functional_results=results),indent=2))
    raise SystemExit(not all(results.values()))
