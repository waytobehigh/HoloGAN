#!/usr/bin/env bash
rsync -azhP --ignore-existing --include='*/' --include='*.png' --include='*.json' --exclude='*' waytobehigh@192.168.1.214:~/Repos/HoloGAN/results .
