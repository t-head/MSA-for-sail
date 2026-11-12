#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

set -euo pipefail

shopt -s nullglob

CACHE_BASE="${MINFER_FMHA_CACHE_DIR:-$HOME/.cache/minfer/msa_for_sail}"

if [[ ! -d "$CACHE_BASE" ]]; then
    echo "cache dir not found: $CACHE_BASE (nothing to clear)"
    exit 0
fi

if [[ "${1:-}" == "--all" ]]; then
    echo "clearing $CACHE_BASE (all msa_for_sail caches)"
    rm -rf "$CACHE_BASE"
    exit 0
fi

for name in sparse_fmha_fwd sparse_fmha_combine sparse_prepare_schedule \
            plan prepare_metadata; do
    dir="$CACHE_BASE/$name"
    if [[ -d "$dir" ]]; then
        rm -rf "$dir"
        echo "cleared $dir"
    else
        echo "skip (absent): $dir"
    fi
done

for dir in "$CACHE_BASE"/sparse_fmha_fwd_* \
           "$CACHE_BASE"/sparse_fmha_combine_* \
           "$CACHE_BASE"/sparse_prepare_schedule_*; do
    rm -rf "$dir"
    echo "cleared $dir"
done

echo "done. next compile will rebuild from source (python -m msa_for_sail.sparse_jit --compile-only all)"
