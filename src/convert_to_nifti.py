"""
OASIS-1 Image Conversion Script
Converts .hdr/.img (Analyze format) → .nii.gz (NIfTI format)
Also parses clinical metadata (CDR, MMSE, Age, Sex, eTIV) into a master CSV.

Author: adapted for OASIS-1 ADNI pipeline replacement
Usage: python convert_to_nifti.py
"""

import os
import glob
import re
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path

# ─── CONFIG ─────────────────────────────────────────────────────────────────
DATA_ROOT   = '../data'                     # folder containing disc1..disc8
OUTPUT_DIR  = '../data/nifti_converted'     # all converted .nii.gz go here
CSV_OUT     = '../data/oasis1_master.csv'   # master metadata CSV
DISCS       = [f'disc{i}' for i in range(1, 9)]   # disc1 … disc8

# We use the skull-stripped, atlas-registered, gain-field-corrected image:
#   *_t88_masked_gfc.img  ← best preprocessed image, already in MNI-like space
IMAGE_PATTERN = 'PROCESSED/MPRAGE/T88_111/*_t88_masked_gfc.img'
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

records = []
errors  = []

for disc in DISCS:
    disc_path = os.path.join(DATA_ROOT, disc)
    if not os.path.isdir(disc_path):
        print(f'[SKIP] {disc} not found')
        continue

    subjects = sorted([
        d for d in os.listdir(disc_path)
        if os.path.isdir(os.path.join(disc_path, d)) and d.startswith('OAS1_')
    ])

    print(f'\n=== {disc}: {len(subjects)} subjects ===')

    for subj in subjects:
        subj_path = os.path.join(disc_path, subj)

        # ── 1. Parse clinical metadata from the .txt file ──────────────────
        txt_file = os.path.join(subj_path, f'{subj}.txt')
        meta = {'ID': subj, 'disc': disc}

        if os.path.exists(txt_file):
            with open(txt_file, 'r') as f:
                content = f.read()
            def extract(key):
                m = re.search(rf'{key}:\s+([^\r\n]+)', content)
                return m.group(1).strip() if m else None

            meta['Age']    = extract('AGE')
            meta['Sex']    = extract('M/F')
            meta['CDR']    = extract('CDR')
            meta['MMSE']   = extract('MMSE')
            meta['eTIV']   = extract('eTIV')
            meta['nWBV']   = extract('nWBV')
            meta['EDUC']   = extract('EDUC')

            # Derive label: CDR=0 or blank → CN, CDR>0 → AD
            try:
                cdr_raw = meta.get('CDR', '') or ''
                cdr_raw = cdr_raw.strip()
                if cdr_raw == '' or cdr_raw == 'None':
                    # Young healthy adult with no CDR recorded → CN
                    meta['label'] = 'CN'
                    meta['CDR'] = 0.0
                else:
                    cdr_val = float(cdr_raw)
                    meta['CDR'] = cdr_val
                    meta['label'] = 'CN' if cdr_val == 0 else 'AD'
            except (TypeError, ValueError):
                meta['label'] = 'CN'   # default to CN if unparseable
                meta['CDR'] = 0.0
        else:
            print(f'  [WARN] No .txt for {subj}')
            meta['label'] = 'Unknown'

        # ── 2. Find the preprocessed .img file ─────────────────────────────
        img_files = glob.glob(os.path.join(subj_path, IMAGE_PATTERN))
        if not img_files:
            errors.append(f'{subj}: No image found')
            print(f'  [ERROR] No image: {subj}')
            continue

        img_path = img_files[0]   # take first if multiple
        hdr_path = img_path.replace('.img', '.hdr')

        if not os.path.exists(hdr_path):
            errors.append(f'{subj}: Missing .hdr')
            print(f'  [ERROR] Missing .hdr: {subj}')
            continue

        # ── 3. Convert to NIfTI ────────────────────────────────────────────
        out_name = f'{subj}.nii.gz'
        out_path = os.path.join(OUTPUT_DIR, out_name)

        if os.path.exists(out_path):
            print(f'  [SKIP] Already converted: {subj}')
            meta['nifti_path'] = out_path
            records.append(meta)
            continue

        try:
            analyze_img = nib.load(img_path)          # nibabel auto-reads .hdr/.img pair
            nifti_img   = nib.Nifti1Image(
                np.array(analyze_img.dataobj),
                affine=analyze_img.affine
            )
            nib.save(nifti_img, out_path)
            meta['nifti_path'] = out_path
            print(f'  [OK]   {subj} → {out_name}  ({meta["label"]})')
        except Exception as e:
            errors.append(f'{subj}: Conversion error — {e}')
            print(f'  [ERROR] {subj}: {e}')
            continue

        records.append(meta)

# ── 4. Save master CSV ────────────────────────────────────────────────────
df = pd.DataFrame(records)
df.to_csv(CSV_OUT, index=False)
print(f'\n✅ Master CSV saved: {CSV_OUT}')
print(f'   Total subjects: {len(df)}')
print(f'   CN: {len(df[df["label"]=="CN"])}  |  AD: {len(df[df["label"]=="AD"])}  |  Unknown: {len(df[df["label"]=="Unknown"])}')
print(f'   NIfTI files saved to: {OUTPUT_DIR}')

if errors:
    print(f'\n⚠️  {len(errors)} errors:')
    for e in errors:
        print(f'   - {e}')
else:
    print('\n✅ No errors.')
