""" Steering file for Geant4 simulations (DD4hep/DDSim). """

from __future__ import absolute_import, unicode_literals
from DDSim.DD4hepSimulation import DD4hepSimulation
from g4units import mm, GeV, radian

# ==============================================================
# User-configurable parameters
# ==============================================================

# Particle definition (PDG name or code)
PARTICLE = "proton"

# Beam energy
ENERGY_GEV = 100

# Source position [mm]
SOURCE_X_MM = 0
SOURCE_Y_MM = 0
SOURCE_Z_MM = 0

# Physics configuration
PHYSICS_LIST = "FTFP_BERT"

# Verbosity level ("INFO", "DEBUG", or integer)
PRINT_LEVEL = "INFO"

# ==============================================================

SIM = DD4hepSimulation()

# ----------------------------------------------------------------
# Run configuration
# ----------------------------------------------------------------

SIM.runType = "run"
SIM.printLevel = PRINT_LEVEL

# ----------------------------------------------------------------
# Particle gun configuration
# ----------------------------------------------------------------

SIM.enableGun   = True
SIM.enableG4Gun = False
SIM.enableG4GPS = False

SIM.crossingAngleBoost = 0.0 * radian
SIM.gun.direction = (0.025, 0.0, 1.0)

# Fixed-energy particle gun
SIM.gun.particle = PARTICLE
SIM.gun.energy   = ENERGY_GEV * GeV
SIM.gun.position = (SOURCE_X_MM, SOURCE_Y_MM, SOURCE_Z_MM)

# Angular distribution (currently disabled)
SIM.gun.isotrop      = False
SIM.gun.distribution = None
# SIM.gun.isotrop      = True
# SIM.gun.distribution = "cos(theta)"
# SIM.gun.thetaMin = 0.0 * radian
# SIM.gun.thetaMax = 0.5 * math.pi * radian
# SIM.gun.phiMin   = 0.0 * radian
# SIM.gun.phiMax   = 2.0 * math.pi * radian

# ----------------------------------------------------------------
# Magnetic field integration
# ----------------------------------------------------------------

SIM.field.stepper             = "HelixSimpleRunge"
SIM.field.largest_step        = 100.0 * mm
SIM.field.min_chord_step      = 1.0e-2 * mm
SIM.field.delta_chord         = 1.0e-3
SIM.field.delta_intersection  = 1.0e-3
SIM.field.delta_one_step      = 0.5e-1 * mm
SIM.field.eps_max             = 1.0e-3
SIM.field.eps_min             = 1.0e-4

# ----------------------------------------------------------------
# Physics list
# ----------------------------------------------------------------

SIM.physics.decays = True
SIM.physics.list   = PHYSICS_LIST

# ----------------------------------------------------------------
# Output, MC truth, and random numbers
# ----------------------------------------------------------------

SIM.output.inputStage = 3
SIM.output.kernel     = 3
SIM.output.part       = 3
SIM.output.random     = 6

SIM.part.enableDetailedHitsAndParticleInfo = False
SIM.part.keepAllParticles                  = True
SIM.part.minDistToParentVertex             = 2.2e-14
SIM.part.printEndTracking                  = False
SIM.part.printStartTracking                = False
SIM.part.saveProcesses                     = ["Decay"]

SIM.random.enableEventSeed = False
SIM.random.file            = None
SIM.random.luxury          = 1
SIM.random.replace_gRandom = True
SIM.random.seed            = None
SIM.random.type            = None