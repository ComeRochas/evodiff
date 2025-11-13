This repo was forked from microsoft/evodiff.

I adapted evodiff/generate.py into evodiff_smg_generate.py where I added the reward functions and the SMC process with multinomial resampling.

You may execute the code from the notebook test_come.ipynb in order to run the SMC guided Protein Generation with multiple configurations. These configs test different values for the SMC frequency (smc_every) and the reward scale (reward_scale) in order to test reward satisfaction VS batch diversity.