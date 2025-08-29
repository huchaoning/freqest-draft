#! /bin/zsh

for f in *.pdf; do
    if [[ -e "$f" ]]; then
        if [[ ! -d gs_optimized ]]; then
            mkdir gs_opt_bak
        fi
        mv "$f" "gs_opt_bak/$f"
        gs -o "$f" -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -dDownsampleColorImages=false  -dDownsampleGrayImages=false -dDownsampleMonoImages=false "gs_opt_bak/$f"
    fi
done
