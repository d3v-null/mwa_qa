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

## Push

```bash
docker push d3vnull0/mwa_qa:latest
```
