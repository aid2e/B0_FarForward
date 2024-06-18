# Cloning this repo

> [!NOTE]
> Working knowledge of `git`, linux terminal, basic `python` is expected to run the code.

* I have my GitHub credentials lined in VSCode and I work within the VS code environemnt. Hence, I simply clone this repo. 
* If directly cloning on non VS code enviroment, Follow the steps in following github personal access tokens from [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
* `git clone https://github.com/aid2e/B0_FarForward.git -b Step-by-step-intro`

# Step by step Instructions to running EPIC EIC Far forward simulations

I am assuming the the entire project is located under the directory in the environment variable `$EIC_PROJECT_DIR`. I have set mine as `export EIC_PROJECT_DIR="/mnt/d/AID2E/Update-FF-Region"`. Also I am using `bash` as my shell.

```
cd $EIC_PROJECT_DIR
```

This is the base working directory for the rest of the rest of the instructions. 

This is what I do in my terminal

![alt text](docs/assests/images/base-dir.png)

## Installation of eic-shell 

* `cd $EIC_PROJECT_DIR`
* `mkdir eic`
* `curl -L https://github.com/eic/eic-shell/raw/main/install.sh | bash`
* `./eic/eic-shell`
* `which ddsim` -- This should output `/opt/local/bin/ddsim`

This is what I get 

![alt text](docs/assests/images/eic-shell.png)

## Installation of epic

* `cd $EIC_PROJECT_DIR`
* If not in eic-shell then `./eic/eic-shell`
* `git clone https://github.com/eic/epic.git -b b0-field-map-testing`
* `cd epic` -- then if you do `git branch` you should be on the branch `b0-field-map-testing`
* Because of recent updates to `epic`, it requires that all the compilations should have no warnings, else it throws in error during compilation. Hence, modify the file `CMakeLists.txt` and comment out the lines `27-29` (See the image below) 

![alt text](docs/assests/images/cmake-epic.png)

* `cd $EIC_PROJECT_DIR`
* If not in eic-shell then `./eic/eic-shell`
* `mkdir epic_install epic_build`
* `cmake -B $EIC_PROJECT_DIR/epic_build -S $EIC_PROJECT_DIR/epic -DCMAKE_INSTALL_PREFIX=$EIC_PROJECT_DIR/epic_install`
* `cmake --build $EIC_PROJECT_DIR/epic_build -j8`
* `cmake --install $EIC_PROJECT_DIR/epic_build`
* `source $EIC_PROJECT_DIR/epic_install/setup.sh`
* `export DETECTOR="epic_craterlake_18x275" && export $DETECTOR_CONFIG=$DETECTOR` -- This sets the correct epic detector geometry. This will be the detector geometry one has to use for Far Forward tasks.

> [!NOTE]
> **👷 TASK ⛑️** : Visualize the detector geometry and take a screenshot of the B0 Detector system. Make sure to point to the correct detector geometry. 
> When I do `dd_web_display $DETECTOR_PATH/${DETECTOR_CONFIG}.xml --export ${DETECTOR_CONFIG}.root` I get the following geometry and figure after zooming.

![alt text](docs/assests/images/B0-detector-system.png)

## Installation of EICrecon

* `cd $EIC_PROJECT_DIR`
* If not in eic-shell then `./eic/eic-shell`
* `source $EIC_PROJECT_DIR/epic_install/setup.sh`
* `git clone https://github.com/eic/EICrecon.git`
* `cd EICrecon` -- then if you do `git branch` you should be on the branch `main`
* `mkdir EICrecon_build EICrecon_install`
* `cmake -B $EIC_PROJECT_DIR/EICrecon_build -S $EIC_PROJECT_DIR/EICrecon -DCMAKE_INSTALL_PREFIX=$EIC_PROJECT_DIR/EICrecon_install`
* `cmake --build $EIC_PROJECT_DIR/EICrecon_build -j8`
* `cmake --install $EIC_PROJECT_DIR/EICrecon_build`
* `source $EIC_PROJECT_DIR/EICrecon_install/bin/eicrecon-this.sh`
* when you do `which eicrecon` it should point to the one in EICrecon_install directory, Mine shows as 

![alt text](docs/assests/images/eic-recon.png)


## Running simulations of protons

Each time one has to source both `epic` and `EICrecon` and set the `$DETECTOR` and `$DETECTOR_CONFIG` before continuing 

* `source $EIC_PROJECT_DIR/epic_install/setup.sh`
* `export DETECTOR="epic_craterlake_18x275" && export $DETECTOR_CONFIG=$DETECTOR`
* `source $EIC_PROJECT_DIR/EICrecon_install/bin/eicrecon-this.sh`
* `cd $EIC_PROJECT_DIR`
* `mkdir $EIC_PROJECT_DIR`