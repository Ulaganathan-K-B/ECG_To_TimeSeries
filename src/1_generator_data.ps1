# --- CONFIGURATION ---
$projectDir = "C:\Users\kbula\PycharmProjects\ECG_Project"
$genScriptDir = "$projectDir\ecg-image-kit\codes\ecg-image-generator"
$originalInput = "$projectDir\ptb-xl"
$stagingInput = "$projectDir\temp_staging_input"

# Temporary folders for the two generation passes
$tempOutputStandard = "$projectDir\temp_raw_output"
$tempOutputBW = "$projectDir\temp_raw_bw_output"

$finalOutput = "$projectDir\training_data_1"
$historyFile = "$projectDir\processed_history.txt"

# --- 1. SETUP & MODE SELECTION ---
Write-Host "--- UNIFIED ECG GENERATION SETUP ---" -ForegroundColor Cyan

# A. Choose User (Determines Complexity & File Assignment)
Write-Host "Select User Identity:"
Write-Host "  [1] User 1 (Simple Mode - 1x Data)"
Write-Host "  [2] User 2 (Simple Mode - 1x Data)"
Write-Host "  [3] User 3 (Complex Mode - 2x Data)"
Write-Host "  [4] User 4 (Complex Mode - 2x Data)"
$UserID = Read-Host "Enter User ID (1-4)"
$UserPrefix = "U$UserID"

# B. Choose Run Type (Test vs Full)
$TestRun = Read-Host "Is this a TEST run? (y/n)"
if ($TestRun -eq 'y') {
    $BatchSize = 5       
    $MaxBatches = 1      
    Write-Host ">> TEST MODE ACTIVE: Will generate 5 images then STOP." -ForegroundColor Yellow
} else {
    $BatchSize = 50      
    $MaxBatches = 999999 
    Write-Host ">> FULL MODE ACTIVE: Processing all assigned files." -ForegroundColor Green
}

# --- 2. DEFINE GENERATOR FLAGS ---
$commonFlags = "-rot 0 --store_config 1 --lead_bbox --lead_name_bbox --print_header --add_qr_code --random_grid_color"

# Standard Augmented Flags (Based on User)
if ($UserID -eq 1 -or $UserID -eq 2) {
    $genFlagsStandard = "$commonFlags --augment -noise 20"
}
elseif ($UserID -eq 3 -or $UserID -eq 4) {
    $genFlagsStandard = "$commonFlags --wrinkles -ca 45 --hw_text -n 4 --x_offset 30 --y_offset 20 --augment -noise 50"
}
else {
    Write-Error "Invalid User ID."
    exit
}

# Black & White Flags (Clean, No Grid)
$genFlagsBW = "--random_grid_present 0 --random_bw 1 --print_header --add_qr_code"


# ---------------------------------------------------------
# 3. CREATE UNIFIED PYTHON ORGANIZER
# ---------------------------------------------------------
$pyScriptContent = @"
import os
import json
import csv
import shutil
import glob
import sys
import time
import wfdb
import pandas as pd
import numpy as np
import re
import cv2

source_dir = sys.argv[1]        # temp_raw_output (Standard)
bw_source_dir = sys.argv[2]     # temp_raw_bw_output (B&W)
dest_base = sys.argv[3]         # training_data
user_prefix = sys.argv[4]       # U1, U2, etc
original_data_dir = sys.argv[5] # temp_staging_input

# Output Folders
img_dest = os.path.join(dest_base, 'images_raw')
bw_dest = os.path.join(dest_base, 'images_bw')
lbl_dest = os.path.join(dest_base, 'labels_coord')
sig_dest = os.path.join(dest_base, 'time_series')

os.makedirs(img_dest, exist_ok=True)
os.makedirs(bw_dest, exist_ok=True)
os.makedirs(lbl_dest, exist_ok=True)
os.makedirs(sig_dest, exist_ok=True)

json_files = glob.glob(os.path.join(source_dir, '**/*.json'), recursive=True)
current_time = int(time.time())
count = 0

print(f'Processing {len(json_files)} generated records...')

for jf in json_files:
    try:
        with open(jf, 'r') as f:
            data = json.load(f)

        # Standard Image paths
        img_src = jf.replace('.json', '.png')
        if not os.path.exists(img_src):
            img_src = jf.replace('.json', '.jpg')
            if not os.path.exists(img_src):
                print(f"Skipping missing standard image for json: {jf}")
                continue

        count += 1
        unique_id = f'{user_prefix}_{current_time}_{count}'

        # --- 1. COORDINATES ---
        csv_rows = []
        if 'leads' in data and isinstance(data['leads'], list):
            for lead in data['leads']:
                lead_name = lead.get('lead_name', 'Unknown')
                bbox_pts = lead.get('lead_bounding_box', {})
                if bbox_pts:
                    xmin = bbox_pts["0"][1]
                    ymin = bbox_pts["0"][0]
                    xmax = bbox_pts["2"][1]
                    ymax = bbox_pts["2"][0]
                    csv_rows.append([lead_name, xmin, ymin, xmax, ymax])

        bbox_path = os.path.join(lbl_dest, f'{unique_id}.csv')
        with open(bbox_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['class', 'xmin', 'ymin', 'xmax', 'ymax'])
            if csv_rows:
                writer.writerows(csv_rows)

        # --- 2. SAVE STANDARD IMAGE ---
        shutil.copy2(img_src, os.path.join(img_dest, f'{unique_id}.png'))

        # --- 3. PROCESS & SAVE B&W IMAGE ---
        base_filename = os.path.splitext(os.path.basename(jf))[0]
        
        # Look for the exact same base filename in the BW directory
        bw_img_src = os.path.join(bw_source_dir, os.path.relpath(img_src, source_dir))
        
        if os.path.exists(bw_img_src):
            img_bw = cv2.imread(bw_img_src)
            if img_bw is not None:
                inverted = cv2.bitwise_not(img_bw)
                cv2.imwrite(os.path.join(bw_dest, f'{unique_id}.png'), inverted)
        else:
            print(f"Warning: Corresponding B&W image not found for {base_filename}")

        # --- 4. SAVE TIME SERIES ---
        original_base = re.sub(r'-\d+$', '', base_filename)
        record_path_root = os.path.join(original_data_dir, original_base)
        header_file = record_path_root + '.hea'

        if os.path.exists(header_file):
            try:
                signals, fields = wfdb.rdsamp(record_path_root)
                lead_names = fields.get('sig_name', [f'Lead_{i}' for i in range(signals.shape[1])])
                df = pd.DataFrame(signals, columns=lead_names)
                df.to_csv(os.path.join(sig_dest, f'{unique_id}.csv'), index=False)
            except Exception as e:
                print(f"Error reading signals for {original_base}: {e}")

    except Exception as e:
        print(f'Critical error processing {jf}: {e}')
"@

$pyScriptPath = "$projectDir\organize_data.py"
Set-Content -Path $pyScriptPath -Value $pyScriptContent -Encoding UTF8

# ---------------------------------------------------------
# 4. ASSIGN FILES
# ---------------------------------------------------------
$allFiles = Get-ChildItem -Path $originalInput -Recurse -Filter "*.dat" | Sort-Object Name
$totalFiles = $allFiles.Count
$shareSize = [math]::Floor($totalFiles / 6)

if ($UserID -eq 1) { $start = 0; $end = $shareSize - 1 }
elseif ($UserID -eq 2) { $start = $shareSize; $end = ($shareSize * 2) - 1 }
elseif ($UserID -eq 3) { $start = ($shareSize * 2); $end = ($shareSize * 4) - 1 }
elseif ($UserID -eq 4) { $start = ($shareSize * 4); $end = $totalFiles - 1 }

$myFiles = $allFiles[$start..$end]
Write-Host "User $UserID assigned range: $start - $end (Total: $($myFiles.Count) source files)" -ForegroundColor Yellow

# ---------------------------------------------------------
# 5. MAIN PROCESSING LOOP
# ---------------------------------------------------------
if (-not (Test-Path $historyFile)) { New-Item $historyFile -ItemType File | Out-Null }
$processedLog = Get-Content $historyFile

# Prepare Staging
if (Test-Path $stagingInput) { Remove-Item $stagingInput -Recurse -Force }
New-Item -ItemType Directory -Path $stagingInput | Out-Null

$currentBatchCount = 0
$batchesCompleted = 0

foreach ($file in $myFiles) {
    if ($processedLog -contains $file.Name) { continue }

    $baseName = $file.BaseName
    $parent = $file.DirectoryName
    Copy-Item "$parent\$baseName.*" -Destination $stagingInput
    $currentBatchCount++

    if ($currentBatchCount -ge $BatchSize) {
        Write-Host "`n=== Processing Batch $($batchesCompleted + 1) ===" -ForegroundColor Cyan
        
        # Clean Temp Dirs
        if (Test-Path $tempOutputStandard) { Remove-Item $tempOutputStandard -Recurse -Force }
        if (Test-Path $tempOutputBW) { Remove-Item $tempOutputBW -Recurse -Force }
        
        cd $genScriptDir

        # PASS 1: Generate Standard Augmented Images
        Write-Host "-> Pass 1: Generating Standard/Augmented Images..." -ForegroundColor Yellow
        $cmdStd = "python gen_ecg_images_from_data_batch.py -i `"$stagingInput`" -o `"$tempOutputStandard`" $genFlagsStandard"
        Invoke-Expression $cmdStd
        
        # PASS 2: Generate Black & White Clean Images
        Write-Host "-> Pass 2: Generating Clean B&W Images..." -ForegroundColor Yellow
        $cmdBW = "python gen_ecg_images_from_data_batch.py -i `"$stagingInput`" -o `"$tempOutputBW`" $genFlagsBW"
        Invoke-Expression $cmdBW
        
        # Organize & Invert
        Write-Host "-> Pass 3: Organizing Data, Extracting Signals, & Inverting B&W..." -ForegroundColor Yellow
        cd $projectDir
        python organize_data.py "$tempOutputStandard" "$tempOutputBW" "$finalOutput" "$UserPrefix" "$stagingInput"

        # Mark done
        $doneFiles = Get-ChildItem -Path $stagingInput -Filter "*.dat"
        foreach ($f in $doneFiles) { Add-Content -Path $historyFile -Value $f.Name }

        Remove-Item "$stagingInput\*" -Recurse -Force
        $currentBatchCount = 0
        $batchesCompleted++

        Write-Host "Batch Saved Successfully." -ForegroundColor Green
        
        if ($batchesCompleted -ge $MaxBatches) {
            Write-Host "Run Complete. Stopping script." -ForegroundColor Magenta
            break
        }
    }
}

# Handle any leftover files
if ($currentBatchCount -gt 0 -and $batchesCompleted -lt $MaxBatches) {
    Write-Host "`n=== Finishing last partial batch... ===" -ForegroundColor Cyan
    if (Test-Path $tempOutputStandard) { Remove-Item $tempOutputStandard -Recurse -Force }
    if (Test-Path $tempOutputBW) { Remove-Item $tempOutputBW -Recurse -Force }
    
    cd $genScriptDir
    
    Write-Host "-> Pass 1: Standard"
    $cmdStd = "python gen_ecg_images_from_data_batch.py -i `"$stagingInput`" -o `"$tempOutputStandard`" $genFlagsStandard"
    Invoke-Expression $cmdStd

    Write-Host "-> Pass 2: B&W"
    $cmdBW = "python gen_ecg_images_from_data_batch.py -i `"$stagingInput`" -o `"$tempOutputBW`" $genFlagsBW"
    Invoke-Expression $cmdBW
    
    Write-Host "-> Pass 3: Organizing & Extracting..."
    cd $projectDir
    python organize_data.py "$tempOutputStandard" "$tempOutputBW" "$finalOutput" "$UserPrefix" "$stagingInput"
    
    $doneFiles = Get-ChildItem -Path $stagingInput -Filter "*.dat"
    foreach ($f in $doneFiles) { Add-Content -Path $historyFile -Value $f.Name }
    Write-Host "All Assigned Files Done!" -ForegroundColor Magenta
}