# manual testing

```bash
bash
source .venv/bin/activate
# python setup.py develop --user
pip install -e .
# export PATH=$PATH:~/.local/bin
```

## from docker image

```bash
docker build -t d3vnull0/mwa_qa:latest .
docker run -it -v /data:/data --entrypoint /bin/bash d3vnull0/mwa_qa:latest
```

## Run Tests

```bash
export obsid=1226233968
export data_dir=/data/dev
export results_dir="${data_dir}/${obsid}/test_results"
[ -d $results_dir ] && rm -rf ${results_dir}/*
mkdir -p $results_dir
chmod a+rwx $results_dir

# prepvis qa
run_prepvisqa.py \
    "${data_dir}/${obsid}/prep/birli_${obsid}_2s_40kHz.uvfits" \
    "${data_dir}/${obsid}/raw/${obsid}.metafits" \
    --out "${results_dir}/birli_${obsid}_prepvis_metrics.json"
plot_prepvisqa.py \
    "${results_dir}/birli_${obsid}_prepvis_metrics.json" --save \
    --out "${results_dir}/birli_${obsid}_prepvis_metrics_modz.png"
# prepvis qa, scoring tiles within cable flavour groups instead of against the
# whole array. use this when a flavour's signal-path gain differs from the
# majority's, which otherwise flags that whole flavour as outliers.
run_prepvisqa.py \
    "${data_dir}/${obsid}/prep/birli_${obsid}_2s_40kHz.uvfits" \
    "${data_dir}/${obsid}/raw/${obsid}.metafits" \
    --cable_flavor \
    --out "${results_dir}/birli_${obsid}_prepvis_metrics_cableflavor.json"
# per-flavour outliers, and the flavour each tile belongs to
jq -r '.XX | to_entries[] | select((.value|type) == "object" and (.value.BAD_ANTS|type) == "array")
    | "\(.key)\t\(.value.BAD_ANTS)"' \
    "${results_dir}/birli_${obsid}_prepvis_metrics_cableflavor.json"
# cal qa
run_calqa.py \
    "${data_dir}/${obsid}/cal/hyp_soln_${obsid}_30l_src4k.fits" \
    "${data_dir}/${obsid}/raw/${obsid}.metafits" \
    --pol X \
    --out "${results_dir}/hyp_soln_${obsid}_30l_src4k_X.json"
plot_calqa.py \
    "${results_dir}/hyp_soln_${obsid}_30l_src4k_X.json" \
     --save --out "${results_dir}/hyp_soln_${obsid}_30l_src4k_X.png"
plot_caljson.py \
    "${results_dir}/hyp_soln_${obsid}_30l_src4k_X.json" \
     --save --out "${results_dir}/hyp_soln_${obsid}_30l_src4k_X_json.png"
# vis qa
run_visqa.py \
    "${data_dir}/${obsid}/cal/hyp_${obsid}_30l_src4k_8s_80kHz.uvfits" \
    --out "${results_dir}/hyp_${obsid}_30l_src4k_8s_80kHz_vis_metrics.json"
plot_visqa.py \
    "${results_dir}/hyp_${obsid}_30l_src4k_8s_80kHz_vis_metrics.json" \
     --save --out "${results_dir}/hyp_${obsid}_30l_src4k_8s_80kHz_vis_metrics_rms.png"
# img qa
run_imgqa.py \
    "${data_dir}/${obsid}/img/wsclean_hyp_${obsid}_30l_src4k_8s_80kHz-MFS-V-dirty.fits" \
    "${data_dir}/${obsid}/img/wsclean_hyp_${obsid}_30l_src4k_8s_80kHz-MFS-XX-dirty.fits" \
    "${data_dir}/${obsid}/img/wsclean_hyp_${obsid}_30l_src4k_8s_80kHz-MFS-YY-dirty.fits" \
    --out "${results_dir}/wsclean_hyp_${obsid}_30l_src4k_8s_80kHz-MFS_img_metrics.json"
plot_imgqa.py \
    "${results_dir}/wsclean_hyp_${obsid}_30l_src4k_8s_80kHz-MFS_img_metrics.json" \
    --save --out "${results_dir}/wsclean_hyp_${obsid}_30l_src4k_8s_80kHz-MFS_img_metrics"
```

test calqa with a dead fits

```bash
run_calqa.py hyp_soln_1324134624.fits 1324134624.metafits --pol X --out hyp_soln_1324134624_X.json
jq -r '.PERCENT_NONCONVERGED_CHS' hyp_soln_1324134624_X.json # should be 100
```

extra

```bash
mkdir -p /cygnus/dev/intfro/nfdata/1340271944/test_results/
run_calqa.py \
    /cygnus/dev/intfro/nfdata/1340271944/cal/hyp_soln_1340271944_30l_src4k_300it.fits \
    /cygnus/dev/intfro/nfdata/1340271944/raw/1340271944.metafits \
    --pol X \
    --out /cygnus/dev/intfro/nfdata/1340271944/test_results/hyp_soln_1340271944_30l_src4k_300it_X.json
```

## Push

```bash
docker push d3vnull0/mwa_qa:latest
```

## Package test data

```bash
export obsid=1226233968
export data_dir=/data/dev
cd "${data_dir}/${obsid}"
zip -r mwa_qa_${obsid}.zip \
    prep/birli_${obsid}_2s_40kHz.uvfits \
    raw/${obsid}.metafits \
    cal/hyp_soln_${obsid}_30l_src4k.fits \
    raw/${obsid}.metafits \
    cal/hyp_${obsid}_30l_src4k_8s_80kHz.uvfits \
    img/wsclean_hyp_${obsid}_30l_src4k_8s_80kHz-MFS-V-dirty.fits \
    img/wsclean_hyp_${obsid}_30l_src4k_8s_80kHz-MFS-XX-dirty.fits \
    img/wsclean_hyp_${obsid}_30l_src4k_8s_80kHz-MFS-YY-dirty.fits
rclone mkdir mwaeor:mwa-qa
rclone copy mwa_qa_${obsid}.zip mwaeor:mwa-qa/ --progress
cd -
```


## Automated tests

`mwa_qa/tests/test_manual_testing.py` runs everything above through the
console scripts. The full observation is 18 GB, so CI uses a trimmed fixture
(a few timesteps and one coarse channel, ~100 MB) published next to it:

```bash
pytest mwa_qa/tests            # downloads the fixture on first run
MWA_QA_DEMO_DATA=/data/dev pytest mwa_qa/tests   # use a full obs instead
MWA_QA_NO_DOWNLOAD=1 pytest mwa_qa/tests         # skip anything needing data
```

Regenerate and republish the fixture after changing what the tests need:

```bash
export obsid=1226233968
scripts/make_demo_fixture.py /data/dev/${obsid} /tmp/fixture/${obsid} --obsid ${obsid}
( cd /tmp/fixture && zip -qr /tmp/mwa_qa_fixture_${obsid}.zip ${obsid} )
rclone copy /tmp/mwa_qa_fixture_${obsid}.zip mwaeor:mwa-qa/ --progress
```

Both live in the same bucket, served at
`https://projects.pawsey.org.au/mwa-qa/`.
