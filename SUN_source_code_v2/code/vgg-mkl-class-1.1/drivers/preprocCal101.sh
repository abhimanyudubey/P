#!/bin/bash

CALDIR=~/data/caltech-101
PREDIR=~/data/caltech-101-prep

echo "Creating directory '$PREDIR'"
mkdir -p "$PREDIR"

echo "Mangling names and reducing very large images"
for i in `cd "$CALDIR" ; find . -name '*.jpg'`
do
    mangled=`echo $i | sed -e 's/\.\///g' -e 's/\//_/g'`
    printf "  C %20s -> %20s\n" "$i" "$mangled"
    #ln -sf "$CALDIR/$i" "$mangled"
    convert -resize '640x640>' "$CALDIR/$i" "$PREDIR/$mangled"
    #convert -resize '480x480' "$CALDIR/$i" "$PREDIR/$mangled"
done
