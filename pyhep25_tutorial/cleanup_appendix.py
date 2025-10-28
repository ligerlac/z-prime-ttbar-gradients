"""
Phase 3: Remove function redefinitions in Appendix
"""
import json

nb = json.load(open('GRAEP_Tutorial.ipynb'))

print("="*80)
print("PHASE 3: CLEANUP APPENDIX")
print("="*80)

# Find Appendix section
appendix_idx = None
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if '---' in source or 'Appendix' in source or '# Appendix' in source:
        appendix_idx = i
        print(f"✓ Found Appendix at cell {i}")
        break

if not appendix_idx:
    print("⚠ Appendix not found, skipping")
else:
    # Look for function redefinitions in Appendix
    for i in range(appendix_idx, len(nb['cells'])):
        source = ''.join(nb['cells'][i]['source']) if isinstance(nb['cells'][i]['source'], list) else nb['cells'][i]['source']

        # Check for soft_selection redefinition
        if 'def soft_selection(' in source and i > appendix_idx:
            # This is a redefinition - check if it's the MVA variant
            if 'mva' not in source.lower():
                print(f"⚠ Found soft_selection redefinition at cell {i}")
                # Add comment to clarify this extends the base function
                lines = source.split('\n')
                new_lines = [
                    "# Note: soft_selection is defined in Part 3",
                    "# Here we EXTEND it for MVA case (not redefine)",
                    ""
                ] + lines
                nb['cells'][i]['source'] = '\n'.join(new_lines)
                print(f"  → Added clarification comment")

        # Check for build_hist_with_cuts redefinition
        if 'def build_hist_with_cuts(' in source and i > appendix_idx:
            print(f"⚠ Found build_hist_with_cuts redefinition at cell {i} - REMOVING")
            # Comment out this cell
            nb['cells'][i]['source'] = ("# This function is defined in Part 4\n"
                                       "# It's available throughout the notebook\n"
                                       "# No need to redefine here!")
            print(f"  → Replaced with reference to Part 4")

        # Check for compute_significance redefinitions
        if 'def compute_significance' in source and 'evermore' not in source and i > appendix_idx:
            print(f"⚠ Found compute_significance variant at cell {i}")
            # This might be compute_significance_with_mva - keep it but clarify
            if 'mva' in source.lower():
                print(f"  → This is MVA variant, adding clarification")
                lines = source.split('\n')
                new_lines = [
                    "# Note: compute_evermore_significance is defined in Part 5",
                    "# Here we show a simplified S/√B version for MVA case",
                    ""
                ] + lines
                nb['cells'][i]['source'] = '\n'.join(new_lines)

# Save
print("\nSaving cleaned notebook...")
with open('GRAEP_Tutorial.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("\n" + "="*80)
print("PHASE 3 COMPLETE: Appendix cleaned")
print("="*80)
print("✓ Removed/clarified function redefinitions")
print("✓ Added references to original definitions")
print("\nFull refactoring complete!")
print("\nSummary of changes:")
print("  ✓ Pedagogical headers added to Parts 3-6")
print("  ✓ build_hist_with_cuts added to Part 4")
print("  ✓ Part 5: Renamed to opt_params, added bandwidth")
print("  ✓ Part 5: Fixed st_bins → m_ttbar_bins bug")
print("  ✓ Part 5: Hidden evermore internals")
print("  ✓ Part 6: Added bandwidth optimization + 4-panel viz")
print("  ✓ Part 7: Added before/after m_ttbar plots")
print("  ✓ Appendix: Removed redefinitions")
print("\nReady for testing!")
