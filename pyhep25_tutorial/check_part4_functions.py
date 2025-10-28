"""
Check what functions are defined in Part 4
"""
import json

nb = json.load(open('GRAEP_Tutorial.ipynb'))

# Find Part 4
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if '## Part 4:' in source:
        print(f"Part 4 starts at cell {i}\n")
        print("="*80)

        # Show next 10 cells
        for j in range(i, min(i + 10, len(nb['cells']))):
            cell_source = ''.join(nb['cells'][j]['source']) if isinstance(nb['cells'][j]['source'], list) else nb['cells'][j]['source']
            first_line = cell_source.split('\n')[0] if cell_source else "(empty)"

            # Check for key patterns
            has_binned_kde = 'def binned_kde_histogram' in cell_source
            has_build_hist = 'def build_hist' in cell_source

            markers = []
            if has_binned_kde:
                markers.append('[BINNED_KDE]')
            if has_build_hist:
                markers.append('[BUILD_HIST]')

            marker_str = ' '.join(markers) if markers else ''

            print(f"Cell {j:2d} ({nb['cells'][j]['cell_type']:8s}): {first_line[:60]:60s} {marker_str}")

        break
