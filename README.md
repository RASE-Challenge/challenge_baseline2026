# Instructions to participants

Please contact us @ rase-challenge@ntu.edu.sg at anytime if you face any issues during the course of this set-up. 

# Download
Download the baseline and training code from this repo. 

```
git clone https://github.com/RASE-Challenge/challenge_baseline2026.git
```

and link dataset from your email after you have registered.


After downloading, unzip the dataset.zip into the repo, and check that the file directory is as such:
```text
challenge_baseline2026/
├── dataset/ # unzip the downloaded dataset and copy here
│   ├── Task1/
│   └── Task2/
├── results/
├── src/
│   ├── config/ 
│   ├── logs/  
│   ├── metrics/ 
│   ├── models/
│   ├── train.py        
│   ├── datamodule.py
│   ├── save_for_submission.py
│   └── utils.py
├── baseline_docker_build.sh
├── baseline_docker_run.sh
├── baseline.dockerfile
├── config.sh
└── README.md
```


## 1. Installing docker on your system

Please follow the quick-start [docker guide](docs/docker_setup_ubuntu.md) that is aimed to help participants with **zero prior Docker experience** install and configure Docker on their system. 


## 2. Running our baseline docker
We have prepared a docker baseline for you to extend (we have prepared a guide for that too). Please make sure that the docker is installed, with [running without sudo enabled](docs/docker_setup_ubuntu.md#4-run-without-sudo-log-out-and-log-back-in-to-take-effect).

Run the build command in challenge_baseline folder via
```bash 
bash baseline_docker_build.sh
```
Followed by the run command
```bash 
bash baseline_docker_run.sh
```

Once successful, you should see 
```bash
root@<container_id>:/src#
```

## 3. VS code set-up (optional)
We recommend using **Visual Studio Code (VS Code)** as your main editor.  
It’s lightweight, cross-platform, and works well with both Python and Docker.
Follow our [VS Code Setup Guide for docker](./docs/vscode_setup.md) (optional).

## 4. Run the training code in the container

Note: Make sure that your command prompt is in root@<container_id>.

Run the following command to train. Note that there are three phases, first with `fast development run` (fast_dev_run), second with `full training`  and finally with `full validation run`.
```bash
bash train.py
```
The rationale for the fast dev run phase is to facilitate quick fails (i.e., code issues, etc).  For now, you can terminate the training script (via `Ctrl+C`) after it has completed the fast development run phase and leverage it's saved file to test the submission portal.

## 5. Submit a dummy evaluation result


To facilitate the submission during the testing phase, we have prepared some trial examples for submitting. Our team believe in protecting your hard work in this challenge and hence, provided a way to protect your IP without requiring you to open-source your submissions. In addition, we provide an single line command to prepare your submission in accordance to the submission platform. 

For submission, we will need to run the following command `inside the docker container`:
```bash
python3 -m pip install pip-chill
```

To submit, just run the following command `inside the docker container`:
```python 
python3 save_for_submission.py -c /results/WaveVoiceNet__learning_rate=0.001_fast_dev_run/fast_dev_run.yaml
```
which will generate the following outputs:
```text
/results/WaveVoiceNet__learning_rate=0.001_fast_dev_run/model_submission.zip
```


With the .zip file, submit it to our Codabench challenge website. The following steps will guide you towards submissions:
1. Register a Codabench account in www.codabench.org (please use the same email registered with your affliation)
2. Visit the challenge website at https://www.codabench.org/competitions/10539/#/.
3. Under "My Submissions" tab, accept the terms and conditions, then register for our challenge.
4. Thereafter, go to the "My submissions" tab, there should be an option of `Trial submission` and `Official submission`. Select `Trial submission` to access if your code is submitting correctly. 

If you have previously registered to us through our registration form, you will be whitelisted which will not require approval.


With the .zip file, you can submit it to our challenge through two methods: website and command line

For UI, in https://www.codabench.org/competitions/10539/#/:
1. Under "My submissions" tab 
2. Check that you have choose to submit as "Yourself" or your organization 
3. Click the clip button on the pop-up window 
4. Attach the zip file, then submit


Please note that since the evaluation is done on our server, we note that bigger models will take substantially longer time to run (please also note that we will only supple a RTX6000 Ada per run). As such, we have restricted everyone to `one official` and `five trial` submissions per day during the validation period and testing period.


## 6. Innovate your new model!
The code base is prepared to allow easy extension. To make your new model, there are only three codes you need to focus on: the model, configuration, and train.py as shown in the following folders/files:
```text
challenge_baseline2026/
├── src/
│   ├── config/train_baseline.yaml  <- copy this and paste in the same folder
│   ├── models/WaveVoiceNet.py <- copy this and paste in the same folder       
│   ├── train.py   <- change config parameter     
│   ├── datamodule.py
│   └── utils.py
├── baseline_docker_build.sh
├── baseline_docker_run.sh
├── baseline.dockerfile
├── config.sh
└── README.md
```

Then, rename `models/WaveVoiceNet copy.py` file to `models/{NewModelName}.py` and within it the `class WaveVoiceNet(BaseModel)` at line 64 to `class {NewModelName}(BaseModel)` where `{NewModelName}` will be your new model name. Then, for `config/train_baseline copy.yaml`, change to `config/train_{NewModelName}`, and change the model parameter as follows:
```yaml
datamodule: 
  length_sec: 4
  batch_size: 8

model: "{NewModelName}" #<-- change this
model_params: 
  learning_rate: 0.001 
```
Finally, in the ```train.py``` file change the following:
```python
# train.py around line 26
OUTPUT_DIR = "/results/" 
CONFIG_FILE = "/src/config/train_{NewModelName}" #<-- change this
```

With the above three, you will be able to start your own modelling.
Make sure that all your model files are encapsulated within the folder **src/models** for smoother submission.



Thank you and enjoy your modelling!





