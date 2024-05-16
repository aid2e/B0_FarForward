echo "SOURCING SETUP"
source epic_install/setup.sh
echo "SOURCING RECON"
source EICrecon_install/bin/eicrecon-this.sh
export DETECTOR=epic_craterlake_18x275

echo DETECTOR=$DETECTOR
echo DETECTOR_VERSION=$DETECTOR_VERSION
echo DETECTOR_CONFIG=$DETECTOR_CONFIG
echo DETECTOR_PATH=$DETECTOR_PATH

export DETECTOR_FILE=${DETECTOR_PATH}/${DETECTOR}.xml

echo JANA_HOME=$JANA_HOME
