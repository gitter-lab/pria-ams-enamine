#!/usr/bin/env bash

target=aldrich
mkdir -p ../datasets/"$target"
python preprocessing4aldrich.py --target="$target" > "$target".out

target=emolecules
mkdir -p ../datasets/"$target"
python preprocessing4aldrich.py --target="$target" > "$target".out
