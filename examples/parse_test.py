import re

import pandas as pd


def extract_nh_rdcs_from_xplor(filepath):
    """Extract N-H RDCs from the 1D3Z XPLOR/CNS restraint file."""
    with open(filepath) as f:
        content = f.read()

    start_idx = content.find('!!! DipolarCouplings.HN-N.tbl\n')
    end_idx = content.find('!!! DipolarCouplings.HN-CO.tbl\n', start_idx)

    rdc_block = content[start_idx:end_idx if end_idx != -1 else len(content)]

    rdc_data = []
    lines = rdc_block.split('\n')

    current_res = None
    for line in lines:
        if 'HN' in line and 'and name' in line:
            parts = line.split(')')
            if len(parts) > 1:
                res_match = re.search(r'resid\s+(\d+)\s+and\s+name', line)
                if res_match:
                    current_res = int(res_match.group(1))
                nums = parts[1].strip().split()
                if nums and current_res is not None:
                    try:
                        val = float(nums[0])
                        rdc_data.append({'res_index': current_res, 'D_exp': val})
                    except ValueError:
                        pass
    return pd.DataFrame(rdc_data).drop_duplicates(subset=['res_index'])

# Test it
df = extract_nh_rdcs_from_xplor('1d3z_mr.str')
print(f"Items: {len(df)}")
print(df.head())
