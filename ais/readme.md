This folder contains an implementation of AIS with q-paths and geometric annealing paths.

### Set up

The implementation is largely based on Xuechen Li's [BDMC package](https://github.com/lxuechen/BDMC).
We first checkout this package in `third` and apply some small changes:

```bash
cd third
git clone git@github.com:lxuechen/BDMC.git
cd BDMC
git checkout 5dfae98a62870a8827afafefc9590c1e3d6c0d2a
git apply ../bdmc.patch
cd ../..
```

This also requires `numpy`, `pytorch` and `tqdm`. These can be installed by using e.g.

```bash
pip install numpy torch tqdm
```

### Examples

We trained a VAE model using the thermodynamic variational objective and stored the model file under `model_files`.
The BDMC gaps on simulated data when using a geometric path and a q-path with `q=0.997` can be obtained by running:

```bash
python run_bdmc_dgm.py \
    --batch-size=50 \
    --no-batches=2 \
    --chain-length=100 \
    --no-samples=10 \
    --seed=1 \
    --model-path=model_files/tvo_model_h_200_z_50_T_100_scdhl_log_lbmin_-1.5_bs_100.pt \
    --geometric

python run_bdmc_dgm.py \
    --batch-size=50 \
    --no-batches=2 \
    --chain-length=100 \
    --no-samples=10 \
    --seed=1 \
    --model-path=model_files/tvo_model_h_200_z_50_T_100_scdhl_log_lbmin_-1.5_bs_100.pt \
    --qpath \
    --q=0.997
```

The results of the quick runs above are:

```
# for the geometric path
average lower bound -133.378, average upper bound -101.352, average bdmc gap: 32.026

# for the q-path with q=0.997
average lower bound -126.694, average upper bound -101.315, average bdmc gap: 25.379
```

The BDMC gaps here are fairly big, but expected given the small number of intermediate densities (100).
In practice, as well as in the paper, we reported results for `chain-length = 500, 1000, 5000` and for a much larger `no-samples`.
