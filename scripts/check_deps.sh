#!/usr/bin/env bash
# Check system dependencies for the OCR pipeline (tesseract, Fraktur tessdata, poppler).
# Usage: bash scripts/check_deps.sh   (exit code 1 if anything is missing)

missing=0

report() {  # report <label> <ok:0|1> <detail>
    if [ "$2" -eq 0 ]; then
        echo "[OK]      $1 $3"
    else
        echo "[MISSING] $1 $4"
        missing=1
    fi
}

# tesseract binary
tess_path=$(which tesseract 2>/dev/null)
report "tesseract" $([ -n "$tess_path" ] && echo 0 || echo 1) "($tess_path)" \
    "- install: brew install tesseract / apt install tesseract-ocr"

# Fraktur tessdata
dirs=()
if [ -n "$tess_path" ]; then
    d=$(tesseract --tessdata-dir 2>/dev/null | head -n1)
    [ -n "$d" ] && dirs+=("$d")
fi
[ -n "$TESSDATA_PREFIX" ] && dirs+=("$TESSDATA_PREFIX" "$TESSDATA_PREFIX/tessdata")
dirs+=(/usr/share/tesseract-ocr/*/tessdata /usr/share/tessdata /usr/local/share/tessdata \
       /opt/homebrew/share/tessdata)
found=""
for d in "${dirs[@]}"; do
    if [ -f "$d/Fraktur.traineddata" ]; then found="$d/Fraktur.traineddata"; break; fi
done
report "Fraktur tessdata" $([ -n "$found" ] && echo 0 || echo 1) "($found)" \
    "- Fraktur.traineddata not found; install tesseract-ocr-script-frak or download it from tessdata_best"

# poppler
pdftoppm_path=$(which pdftoppm 2>/dev/null)
report "poppler (pdftoppm)" $([ -n "$pdftoppm_path" ] && echo 0 || echo 1) "($pdftoppm_path)" \
    "- install: brew install poppler / apt install poppler-utils"

# optional engines for the benchmark (scripts/ocr_benchmark.py); missing ones do not fail the check
optional() {  # optional <label> <binary>
    p=$(which "$2" 2>/dev/null)
    if [ -n "$p" ]; then echo "[OK]      $1 ($p)"; else echo "[OPTIONAL] $1 not found - $3"; fi
}
optional "kraken" "${KRAKEN_BIN:-kraken}" "pip install kraken (own venv), or set KRAKEN_BIN"
optional "calamari" "${CALAMARI_BIN:-calamari-predict}" "pip install calamari-ocr (own venv), or set CALAMARI_BIN"
python3 -c "import pytesseract" 2>/dev/null && echo "[OK]      pytesseract" \
    || { echo "[MISSING] pytesseract - pip install -r requirements.txt"; missing=1; }

exit $missing
