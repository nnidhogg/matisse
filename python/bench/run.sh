#!/bin/sh
# Runs the collection docs/python.md's figures are read from: matisse's C++ bench, matisse_bench, and this directory's
# bench.py for the Python package and tiktoken, on the same corpora, one after another, each output kept whole.
#
#     sh run.sh MATISSE_BENCH PYTHON VOCAB CORPORA LARGE OUTPUT_DIR
#
# MATISSE_BENCH is a Release build's tools/probes/matisse_bench, PYTHON an interpreter with matisse-tokenizer, tiktoken
# and numpy installed, VOCAB the pinned vocabulary files, CORPORA the corpora of docs/benchmarks.md, LARGE the directory
# holding enwik8, and OUTPUT_DIR where the outputs go. The single-thread rows run under taskset -c 2, as
# docs/benchmarks.md's do, and the threaded rows unpinned.

set -eu

bench=$1
python=$2
vocab=$3
corpora=$4
large=$5
out=$6
here=$(cd "$(dirname "$0")" && pwd)
all=gpt2,cl100k,o200k,llama3

mkdir -p "$out"

{
    date -u '+%Y-%m-%d %H:%M:%S UTC'
    git -C "$here" rev-parse --short HEAD
    git -C "$here" status --short | head -20
    uname -srm
    lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket'
    free -g | head -2
    "$python" --version
    g++ --version | head -1
} > "$out/machine.txt"

"$python" -I "$here/bench.py" short "$corpora/prompts.txt" "$out/short.txt" 48

for corpus in gutenberg wiki-zh wiki-ru; do
    taskset -c 2 "$bench" single "$vocab" $all 5 "$corpora/$corpus.txt" > "$out/cpp-single-$corpus.txt"
    taskset -c 2 "$python" -I "$here/bench.py" single "$vocab" $all 5 "$corpora/$corpus.txt" \
        > "$out/py-single-$corpus.txt"
done

taskset -c 2 "$bench" single "$vocab" $all 3 "$large/enwik8" > "$out/cpp-single-enwik8.txt"
taskset -c 2 "$python" -I "$here/bench.py" single "$vocab" $all 3 "$large/enwik8" > "$out/py-single-enwik8.txt"

for prompts in "$corpora/prompts.txt" "$out/short.txt"; do
    name=$(basename "$prompts")
    taskset -c 2 "$bench" prompts "$vocab" $all 5 "$prompts" > "$out/cpp-prompts-$name"
    taskset -c 2 "$python" -I "$here/bench.py" prompts "$vocab" $all 5 "$prompts" > "$out/py-prompts-$name"
done

"$bench" parallel "$vocab" $all 3 "$large/enwik8" 1 2 4 8 12 16 24 > "$out/cpp-parallel-enwik8.txt"
"$python" -I "$here/bench.py" parallel "$vocab" $all 3 "$large/enwik8" 1 2 4 8 12 16 24 > "$out/py-parallel-enwik8.txt"

"$bench" batch "$vocab" $all 3 "$corpora/prompts.txt" 1 24 > "$out/cpp-batch-prompts.txt"
"$python" -I "$here/bench.py" batch "$vocab" $all 3 "$corpora/prompts.txt" 1 24 > "$out/py-batch-prompts.txt"
"$python" -I "$here/bench.py" threads "$vocab" $all 3 "$corpora/prompts.txt" 1 24 > "$out/py-threads-prompts.txt"

"$python" -I "$here/bench.py" load "$vocab" $all 7 > "$out/py-load.txt"

rm "$out/short.txt"
