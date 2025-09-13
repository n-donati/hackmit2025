from typing import Dict, Any, List


def build_strip_final(values_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a stable, structured final JSON for strip analysis from the
    values mapping returned by process_image().
    """
    analytes: List[Dict[str, Any]] = []
    for test_name, entry in (values_dict or {}).items():
        analytes.append({
            'name': test_name,
            'value': entry.get('value') if isinstance(entry, dict) else entry,
        })

    final_obj = {
        'type': 'strip_analysis',
        'analytes': analytes,
        'num_analytes': len(analytes),
        'values': values_dict or {},
    }
    return final_obj


