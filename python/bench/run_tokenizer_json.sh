#!/bin/sh
# Runs the collection of docs/python.md's figures beside Hugging Face tokenizers: matisse-tokenizer and Hugging Face's
# tokenizers package reading the same pinned tokenizer.json files, through this directory's bench.py, on the corpora
# of docs/benchmarks.md, the ids of every pair compared before any of it is timed, each output kept whole.
#
#     sh run_tokenizer_json.sh PYTHON VOCAB CORPORA LARGE OUTPUT_DIR
#
# PYTHON is an interpreter with matisse-tokenizer, tokenizers, tiktoken and numpy installed, VOCAB the pinned vocabulary
# files with the four tokenizer.json files, CORPORA the corpora of docs/benchmarks.md, LARGE the directory holding
# enwik8, and OUTPUT_DIR where the outputs go. The single-thread rows run under taskset -c 2 with RAYON_NUM_THREADS=1,
# and the batch rows unpinned with RAYON_NUM_THREADS set to their thread count. The runs on enwik8 raise the soft limit
# on the address space to the hard limit, since Hugging Face's Encoding of enwik8 takes 14 GiB. Each timed run is one
# tokenizer's, so that a run disturbed by other work on the machine can be repeated alone.

set -eu

python=$1
vocab=$2
corpora=$3
large=$4
out=$5
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
    echo "ulimit -v: soft $(ulimit -S -v), hard $(ulimit -H -v)"
    "$python" --version
    "$python" -I -c 'import importlib.metadata as m; print("\n".join(sorted(f"{d.name}=={d.version}"
for d in m.distributions())))'
    for tokenizer in gpt2 cl100k o200k llama3; do
        sha256sum "$vocab/$tokenizer-tokenizer.json" | sed "s|$vocab/||"
    done
} > "$out/machine.txt"

"$python" -I "$here/bench.py" short "$corpora/prompts.txt" "$out/short.txt" 48

"$python" -I "$here/bench.py" json-digests "$vocab" $all "$corpora/twitter.json" "$corpora/munch-python@15b2863c.txt" \
    "$corpora/munch-cpp@15b2863c.txt" "$corpora/gutenberg.txt" "$corpora/wiki-zh.txt" "$corpora/wiki-ar.txt" \
    "$corpora/wiki-ru.txt" "$corpora/wiki-hi.txt" "$corpora/wiki-sv.txt" "$corpora/prompts.txt" "$out/short.txt" \
    > "$out/json-digests.txt"
(ulimit -S -v "$(ulimit -H -v)" && "$python" -I "$here/bench.py" json-digests "$vocab" $all "$large/enwik8") \
    > "$out/json-digests-enwik8.txt"

for tokenizer in gpt2 cl100k o200k llama3; do
    for corpus in gutenberg wiki-zh wiki-ru; do
        RAYON_NUM_THREADS=1 taskset -c 2 "$python" -I "$here/bench.py" json-single "$vocab" $tokenizer 5 \
            "$corpora/$corpus.txt" > "$out/json-single-$corpus-$tokenizer.txt"
    done
    (ulimit -S -v "$(ulimit -H -v)" && RAYON_NUM_THREADS=1 taskset -c 2 "$python" -I "$here/bench.py" json-single \
        "$vocab" $tokenizer 3 "$large/enwik8") > "$out/json-single-enwik8-$tokenizer.txt"
    for prompts in "$corpora/prompts.txt" "$out/short.txt"; do
        RAYON_NUM_THREADS=1 taskset -c 2 "$python" -I "$here/bench.py" json-prompts "$vocab" $tokenizer 5 "$prompts" \
            > "$out/json-prompts-$(basename "$prompts" .txt)-$tokenizer.txt"
    done
    for threads in 1 24; do
        RAYON_NUM_THREADS=$threads "$python" -I "$here/bench.py" json-batch "$vocab" $tokenizer 3 \
            "$corpora/prompts.txt" "$threads" > "$out/json-batch-prompts-x$threads-$tokenizer.txt"
    done
done

rm "$out/short.txt"
