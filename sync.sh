#!/usr/bin/env bash
rsync -azhP --ignore-existing --include='*/' --include='*.png' --include='*.json' --exclude='*' $1/HoloGAN/results .
