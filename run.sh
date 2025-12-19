#!/bin/bash
# Vesuvius Challenge - Inverse Hensel Scroll Unwrapper
# One-click script to regenerate all deliverables

set -e

echo "=============================================="
echo "VESUVIUS INVERSE HENSEL SCROLL UNWRAPPER"
echo "=============================================="

OUTPUT_DIR="${1:-./outputs}"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Processing both scrolls..."
echo ""

python inverse_hensel_unwrapper.py --all --output "$OUTPUT_DIR"

echo ""
echo "=============================================="
echo "DELIVERABLES GENERATED"
echo "=============================================="
echo ""
ls -lh "$OUTPUT_DIR"
echo ""
echo "Human hours: 0 (fully automatic)"
