"""Pinned ACE XML numeric projection; Java/process/filesystem details live elsewhere."""
from __future__ import annotations
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def read_definitions(path: Path):
    root=ET.parse(path).getroot()
    if root.tag != 'feature_key_file':
        raise ValueError('unexpected ACE definitions root')
    result=[]
    for feature in root.findall('feature'):
        name=feature.findtext('name')
        dimensions=int(feature.findtext('parallel_dimensions','0'))
        if not name or dimensions<1 or name in {f['name'] for f in result}:
            raise ValueError('invalid/duplicate ACE feature definition')
        result.append(dict(name=name,dimensions=dimensions))
    return result


def numeric_schema(definitions):
    return [dict(name=f['name']+f'[{i:03d}]',backend='jsymbolic_2.2',unit='upstream_defined',nullable=True)
            for f in definitions for i in range(f['dimensions'])]


def parse_exports(values: Path, definitions: Path, expected: list[dict]):
    actual=read_definitions(definitions)
    by_name={f['name']:f for f in actual}
    if any(by_name.get(f['name']) != f for f in expected):
        raise ValueError('pinned definition schema mismatch')
    root=ET.parse(values).getroot()
    if root.tag != 'feature_vector_file':
        raise ValueError('unexpected ACE values root')
    datasets=root.findall('data_set')
    if len(datasets)!=1 or datasets[0].findall('section'):
        raise ValueError('expected exactly one whole-piece ACE row')
    row={}
    for feature in datasets[0].findall('feature'):
        name=feature.findtext('name')
        if not name or name in row:
            raise ValueError('duplicate/unnamed ACE feature')
        row[name]=[v.text or '' for v in feature.findall('v')]
    projected=[]
    diagnostics=dict(missing=[],nonfinite=[],excluded=sorted(set(row)-{f['name'] for f in expected}))
    for feature in expected:
        name=feature['name']; dimension=feature['dimensions']
        values_for_feature=row.get(name)
        if values_for_feature is None:
            projected.extend([None]*dimension)
            diagnostics['missing'].append(name)
            continue
        if len(values_for_feature)!=dimension:
            raise ValueError('ACE value dimension mismatch: '+name)
        for index,value in enumerate(values_for_feature):
            if value.strip().lower() in ('','null','nan','n/a','na'):
                projected.append(None); diagnostics['missing'].append(name+f'[{index:03d}]')
            else:
                number=float(value)  # nonnumeric/locale-corrupt fields are failures, never labels
                if math.isfinite(number): projected.append(number)
                else:
                    projected.append(None); diagnostics['nonfinite'].append(name+f'[{index:03d}]')
    if not any(v is not None for v in projected):
        raise ValueError('no finite pinned musical features')
    return dict(schema=numeric_schema(expected),values=projected,diagnostics=diagnostics)
