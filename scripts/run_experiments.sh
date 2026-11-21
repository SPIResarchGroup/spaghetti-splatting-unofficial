#!/usr/bin/env bash
# All experiments (30k iterations each) on GPUs 4-7:
#   GPU4: bonsai_gauss  + synth ablations (no split, no clone)
#   GPU5: bonsai_beta
#   GPU6: bonsai_points + synth ablations (no merge, no attraction)
#   GPU7: synth_gauss, synth_beta, synth_points
# then strand evaluation of all synthetic runs.  Usage: bash scripts/run_experiments.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}  # CUDA_HOME: optional, auto-detected (needs an nvcc matching your PyTorch / GPU)
mkdir -p logs runs

run() {  # gpu name args...
  local gpu=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu $PY train.py --out runs/$name "$@" > logs/$name.log 2>&1
}

B="--data data/360_v2/bonsai --images images_2 --cap ${CAP:-1000000}"
S="--data data/synth_spaghetti --type synthetic --cap 100000"

run 4 bonsai_gauss  $B --kernel gaussian &
run 5 bonsai_beta   $B --kernel beta &
run 6 bonsai_points $B --points &
run 7 synth_gauss   $S --kernel gaussian &
run 7 synth_beta    $S --kernel beta &
run 7 synth_points  $S --points &
( run 4 synth_abl_nosplit   $S --no_split;   run 4 synth_abl_noclone   $S --no_clone ) &
( run 6 synth_abl_nomerge   $S --no_merge;   run 6 synth_abl_noattract $S --no_attract ) &
wait
for r in synth_gauss synth_beta synth_points synth_abl_nosplit synth_abl_noclone synth_abl_nomerge synth_abl_noattract; do
  $PY scripts/eval_strands.py --run runs/$r > logs/${r}_strands.log 2>&1
done
$PY scripts/summarize.py > results.md
echo ALL_DONE
