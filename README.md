# SSELFI
Summary Statistics Estimation and Likelihood-Free Inference


## Setting up on the Google Cloud

See instructions on this page: https://cloud.google.com/tpu/docs/quickstart

and run the following command in the console:
```
$ ctpu up -zone us-central1-c
```

This  will create a VM instance and a TPU. Be careful, you are being charge as long as the TPU and VM are running, use  the following to stop:
```
$ ctpu pause
```


## Training
python sselfi_main.py --data_dir=gs://sselfi-camelus/training --model_dir=gs://sselfi-camelus/models/test2 --export_dir=gs://selfi-camelus/exports/test2
