"""
Generate form data for the website prediction form
Extracts unique values for categorical features and ranges for numeric features
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load the metadata
meta_path = Path("../data/meta_for_model_MiniLM-L6-v2.parquet")
df = pd.read_parquet(meta_path)

print(f"Loaded {len(df)} records")
print(f"Columns: {df.columns.tolist()}")

# Extract feature information
form_data = {}

# Numeric feature: objectEndDate
if 'objectEndDate' in df.columns:
    form_data['objectEndDate'] = {
        "type": "numeric",
        "min": float(df['objectEndDate'].min()),
        "max": float(df['objectEndDate'].max()),
        "median": float(df['objectEndDate'].median()),
        "mean": float(df['objectEndDate'].mean()),
        "description": "Year when the artwork was completed"
    }

# Categorical features
categorical_features = {
    'department': "Department",
    'country': "Country of origin",
    'cat1': "Primary category",
    'subcat1': "Subcategory (classification)",
    'cat2': "Secondary category (object type)"
}

for col, description in categorical_features.items():
    if col in df.columns:
        # Get unique values, sorted, excluding None/NaN
        unique_vals = df[col].dropna().unique().tolist()
        unique_vals = [str(v) for v in unique_vals if str(v) != 'nan']
        unique_vals = sorted(set(unique_vals))
        
        form_data[col] = {
            "type": "categorical",
            "options": unique_vals,
            "count": len(unique_vals),
            "description": description,
            "most_common": df[col].value_counts().head(10).index.tolist()
        }
        
        print(f"\n{col}: {len(unique_vals)} unique values")
        print(f"  Most common: {form_data[col]['most_common'][:5]}")

# Text field info
form_data['text'] = {
    "type": "text",
    "description": "Combined text description of the artwork (title, artist, culture, medium, etc.)",
    "example": "Oil painting depicting a pastoral landscape with figures from French countryside",
    "min_length": 10,
    "max_length": 1000,
    "required": True
}

# Save to JSON
output_path = Path("form_config.json")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(form_data, f, indent=2, ensure_ascii=False)

print(f"\n[OK] Saved form configuration to {output_path}")

# Also create a simplified version for dropdowns (just the options)
simplified = {
    'objectEndDate': {
        'min': form_data['objectEndDate']['min'],
        'max': form_data['objectEndDate']['max'],
        'median': form_data['objectEndDate']['median']
    }
}

for col in ['department', 'country', 'cat1', 'subcat1', 'cat2']:
    if col in form_data:
        simplified[col] = form_data[col]['options']

output_simple = Path("form_options.json")
with open(output_simple, 'w', encoding='utf-8') as f:
    json.dump(simplified, f, indent=2, ensure_ascii=False)

print(f"[OK] Saved simplified options to {output_simple}")
