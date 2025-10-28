import json

nb = json.load(open('GRAEP_Tutorial.ipynb'))

print("CURRENT NOTEBOOK STRUCTURE:\n")
print("="*80)

for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    first_line = source.split('\n')[0] if source else "(empty)"

    # Check for key definitions
    if cell['cell_type'] == 'code':
        has_soft_selection = 'def soft_selection' in source
        has_build_hist = 'def build_hist' in source or 'def binned_kde' in source
        has_evermore = 'import evermore' in source
        has_compute_sig = 'def compute_evermore_significance' in source or 'def compute_significance' in source

        markers = []
        if has_soft_selection:
            markers.append('[SOFT_SEL]')
        if has_build_hist:
            markers.append('[HIST]')
        if has_evermore:
            markers.append('[EVM]')
        if has_compute_sig:
            markers.append('[SIG]')

        marker_str = ' '.join(markers) if markers else ''

        if len(first_line) > 60:
            first_line = first_line[:60] + "..."
        print(f"Cell {i:2d} ({cell['cell_type']:8s}): {first_line:65s} {marker_str}")
    else:
        if len(first_line) > 70:
            first_line = first_line[:70] + "..."
        print(f"Cell {i:2d} ({cell['cell_type']:8s}): {first_line}")

print(f"\nTotal cells: {len(nb['cells'])}")
