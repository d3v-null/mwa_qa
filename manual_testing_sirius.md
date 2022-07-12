## Build docker image

```bash
docker build -t d3vnull0/mwa_qa:latest .
```

## Vis QA

generate test files with

```bash
export prefix="/data/dev/1090008640"
cd prefix

```

```bash
# export prefix="/data/dev/pretty/1214752992/1214752992_trunc"
# export prefix="/data/dev/1257518400/prep/1257518400"
export prefix="/data/dev/1254670392_vis/1254670392"
[ -f "${prefix}_vis_metrics.json" ] && rm -rf "${prefix}_vis_metrics.json"
docker run -it -v /data:/data --entrypoint run_visqa.py d3vnull0/mwa_qa:latest "${prefix}.uvfits"
ls -al "${prefix}_vis_metrics.json"
```

## Cal QA

```bash
export prefix="/home/dev/mwa_qa/mwa_qa/data/hyp_soln_1060539904_poly_30l_src4k"
[ -f "${prefix}_cal_metrics.json" ] && rm -rf "${prefix}_cal_metrics.json"
docker run -it -v "$(pwd)/mwa_qa/data:/data" --entrypoint run_calqa.py d3vnull0/mwa_qa:latest \
    "${prefix}.fits" "/data/1060539904.metafits"
ls -al "${prefix}_cal_metrics.json"
```

## Push

```bash
docker push d3vnull0/mwa_qa:latest
```
