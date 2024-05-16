# B0_FarForward

Currently the code only supports local optimization. One has to be within the Docker container in order to run optimization

## How to run simulations

First make sure to first install eic shell and checkout the branch 

### eic-shell

* `mkdir -p path/to/install/eicshell/eic`
* `cd /path/to/install/eicshell/eic`
* `curl -L https://github.com/eic/eic-shell/raw/main/install.sh | bash`
* `./eic-shell`

### epic installation

* `git clone https://github.com/eic/epic.git -b b0-field-map-testing`
* `mkdir epic_build epic_install`
* `cd epic`
* `cmake -B ../epic_build -S . -DCMAKE_INSTALL_PREFIX=../epic_install`
* `cmake --build ../epic_build --target install`
* `cd ..`

### EIC Recon installation

* `git clone https://github.com/eic/EICrecon.git`
* `mkdir EICrecon_build EICrecon_install`
* `cd EICrecon`
* `cmake -B ../EICrecon_build -S . -DCMAKE_INSTALL_PREFIX=../EICrecon_install`
* `cmake --build ../EICrecon_build --target install`
* `cd ..`

### Sourcing it

* `cd B0_FarForward`
* `source ff_work.sh`

## Running simulations

After sourcing in order to run simulation in a multi processing fashion change directory to `cd ProjectUtils`

```python RunSimulations.py --steeringFile /sciclone/data10/ksuresh/AID2E/FarForward/FromAlex/ddsim_steer_B0_testing.py --parentDir $PWD --runDir $PWD/Directory/to/run/simulations```

THe script will split the total number of events into the number of threads/cores given and will create subdirectories and run simulation using joblib's multiprocessing library

## Running optimization (mobo)

Make sure to be inside of eic shell

```python wrapper_mobo.py -c optimize.config```


