#!/bin/bash

RUNNING_DIR=$PWD
echo "Starting to perform simulation in ${RUNNING_DIR}" 

EPIC_INSTALL=epic_install
EICRECON_INSTALL=eicrecon_install
CODE_DIR=code_dir
OUT_DIR=out_dir
N_EVENTS=n_events

source $EPIC_INSTALL/setup.sh
source $EICRECON_INSTALL/bin/eicrecon-this.sh

export DETECTOR="epic_craterlake_18x275_default_interlayer_32cm"
export DETECTOR_CONFIG="epic_craterlake_18x275_default_interlayer_32cm"

compactFile=${DETECTOR_PATH}/${DETECTOR_CONFIG}.xml

echo "Running npsim simulation"
npsim --steering $CODE_DIR/ddsim_steer_B0_testing.py --numberOfEvents $N_EVENTS --compactFile $compactFile --outputFile ${DETECTOR}.edm4hep.root > $PWD/sim_log.out 2>sim_log.err

echo "Running reconstruction"
eicrecon -Pdd4hep:xml_files=$compactFile -Ppodio_output_include_collections=ReconstructedParticles,GeneratedParticles,ReconstructedChargedParticles,B0TrackerRecHits ${DETECTOR}.edm4hep.root > recon_out.log 2>recon_err.log

echo "analyzing output"
root -q -b ''${CODE_DIR}'/SimpleAnalysis.C("podio_output.root")' > ana_out.log 2>ana_err.log


