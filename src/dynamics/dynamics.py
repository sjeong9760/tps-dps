# We refer to https://github.com/LarsHoldijk/SOCTransitionPaths/blob/master/potentials/alanine_md.py
# for applying neural network bias force to OpenMM

import openmm as mm
from openmm import app
import openmm.unit as unit
from openmmtools.integrators import VVVRIntegrator

from .base import BaseDynamics


class Aldp(BaseDynamics):
    def __init__(self, args, state):
        super().__init__(args, state)

    def setup(self):
        forcefield = app.ForceField("amber99sbildn.xml")
        pdb = app.PDBFile(self.start_file)
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            ewaldErrorTolerance=0.0005,
        )
        external_force = mm.CustomExternalForce("-(fx*x+fy*y+fz*z)")
        external_force.addPerParticleParameter("fx")
        external_force.addPerParticleParameter("fy")
        external_force.addPerParticleParameter("fz")
        system.addForce(external_force)
        for i in range(len(pdb.positions)):
            external_force.addParticle(i, [0, 0, 0])
        integrator = VVVRIntegrator(
            self.temperature,
            self.friction,
            self.timestep,
        )
        integrator.setConstraintTolerance(0.00001)
        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        return pdb, integrator, simulation, external_force


class Chignolin(BaseDynamics):
    def __init__(self, args, state):
        super().__init__(args, state)

    def setup(self):
        forcefield = app.ForceField(
            "data/protein.ff14SBonlysc.xml", "implicit/gbn2.xml"
        )
        pdb = app.PDBFile(self.start_file)
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            ewaldErrorTolerance=0.0005,
        )
        external_force = mm.CustomExternalForce("-(fx*x+fy*y+fz*z)")
        external_force.addPerParticleParameter("fx")
        external_force.addPerParticleParameter("fy")
        external_force.addPerParticleParameter("fz")
        system.addForce(external_force)
        for i in range(len(pdb.positions)):
            external_force.addParticle(i, [0, 0, 0])
        integrator = VVVRIntegrator(
            self.temperature,
            self.friction,
            self.timestep,
        )
        integrator.setConstraintTolerance(0.00001)
        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        return pdb, integrator, simulation, external_force


class Trpcage(BaseDynamics):
    def __init__(self, args, state):
        super().__init__(args, state)

    def setup(self):
        forcefield = app.ForceField(
            "data/protein.ff14SBonlysc.xml", "implicit/gbn2.xml"
        )
        pdb = app.PDBFile(self.start_file)
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            ewaldErrorTolerance=0.0005,
        )
        external_force = mm.CustomExternalForce("-(fx*x+fy*y+fz*z)")
        external_force.addPerParticleParameter("fx")
        external_force.addPerParticleParameter("fy")
        external_force.addPerParticleParameter("fz")
        system.addForce(external_force)
        for i in range(len(pdb.positions)):
            external_force.addParticle(i, [0, 0, 0])
        integrator = VVVRIntegrator(
            self.temperature,
            self.friction,
            self.timestep,
        )
        integrator.setConstraintTolerance(0.00001)
        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        return pdb, integrator, simulation, external_force


class Bba(BaseDynamics):
    def __init__(self, args, state):
        super().__init__(args, state)

    def setup(self):
        forcefield = app.ForceField(
            "data/protein.ff14SBonlysc.xml", "implicit/gbn2.xml"
        )
        pdb = app.PDBFile(self.start_file)
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            ewaldErrorTolerance=0.0005,
        )
        external_force = mm.CustomExternalForce("-(fx*x+fy*y+fz*z)")
        external_force.addPerParticleParameter("fx")
        external_force.addPerParticleParameter("fy")
        external_force.addPerParticleParameter("fz")
        system.addForce(external_force)
        for i in range(len(pdb.positions)):
            external_force.addParticle(i, [0, 0, 0])
        integrator = VVVRIntegrator(
            self.temperature,
            self.friction,
            self.timestep,
        )
        integrator.setConstraintTolerance(0.00001)
        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        return pdb, integrator, simulation, external_force


class Bbl(BaseDynamics):
    def __init__(self, args, state):
        super().__init__(args, state)

    def setup(self):
        forcefield = app.ForceField(
            "data/protein.ff14SBonlysc.xml", "implicit/gbn2.xml"
        )
        pdb = app.PDBFile(self.start_file)
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            ewaldErrorTolerance=0.0005,
        )
        external_force = mm.CustomExternalForce("-(fx*x+fy*y+fz*z)")
        external_force.addPerParticleParameter("fx")
        external_force.addPerParticleParameter("fy")
        external_force.addPerParticleParameter("fz")
        system.addForce(external_force)
        for i in range(len(pdb.positions)):
            external_force.addParticle(i, [0, 0, 0])
        integrator = VVVRIntegrator(
            self.temperature,
            self.friction,
            self.timestep,
        )
        integrator.setConstraintTolerance(0.00001)
        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        return pdb, integrator, simulation, external_force


class Poly(BaseDynamics):
    def __init__(self, args, state):
        super().__init__(args, state)

    def setup(self):
        forcefield = app.ForceField(
            "data/protein.ff14SBonlysc.xml", "implicit/gbn2.xml"
        )
        pdb = app.PDBFile(self.start_file)
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            rigidWater=True,
            ewaldErrorTolerance=0.0005,
        )
        external_force = mm.CustomExternalForce("-(fx*x+fy*y+fz*z)")
        external_force.addPerParticleParameter("fx")
        external_force.addPerParticleParameter("fy")
        external_force.addPerParticleParameter("fz")
        system.addForce(external_force)
        for i in range(len(pdb.positions)):
            external_force.addParticle(i, [0, 0, 0])
        integrator = VVVRIntegrator(
            self.temperature,
            self.friction,
            self.timestep,
        )
        integrator.setConstraintTolerance(0.00001)
        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        return pdb, integrator, simulation, external_force

class G4(BaseDynamics):
    def __init__(self, args, state):
        super().__init__(args, state)

    def setup(self):
        forcefield = app.ForceField(
            "data/g4/DNA.bsc1.xml", "implicit/gbn2.xml"
        )
        pdb = app.PDBFile(self.start_file)
        newTopology, template = self._initialize_topology(pdb.topology)
        system = forcefield.createSystem(
            newTopology,
            nonbondedMethod=app.NoCutoff,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            rigidWater=True,
            ewaldErrorTolerance=0.0005,
            residueTemplates=template
        )
        external_force = mm.CustomExternalForce("-(fx*x+fy*y+fz*z)")
        external_force.addPerParticleParameter("fx")
        external_force.addPerParticleParameter("fy")
        external_force.addPerParticleParameter("fz")
        system.addForce(external_force)
        for i in range(len(pdb.positions)):
            external_force.addParticle(i, [0, 0, 0])
        integrator = VVVRIntegrator(
            self.temperature,
            self.friction,
            self.timestep,
        )
        integrator.setConstraintTolerance(0.00001)

        #platform = mm.Platform.getPlatformByName('CUDA')   # Use NVIDIA GPU
        #properties = {'CudaPrecision': 'mixed'}
        #properties["DeviceIndex"] = "0"

        #simulation = app.Simulation(newTopology, system, integrator, platform, properties)
        simulation = app.Simulation(newTopology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        return pdb, integrator, simulation, external_force
    
    def _initialize_topology(self, topology):
        def gen_newTop(top):
            chain0 = [chain for chain in top.chains() if chain.index == 0][0]
            base_idx = [res.index for res in chain0.residues()]
            start_idx, end_idx = base_idx[0], base_idx[-1]

            boxvec = top.getPeriodicBoxVectors()._value
            newTopology = app.Topology()
            newTopology.setPeriodicBoxVectors((boxvec[0],boxvec[1],boxvec[2])*unit.nanometer)
            for chain in top.chains():
                newChain = newTopology.addChain()
                for res in chain.residues():
                    if res.name.endswith('A') and res.index == start_idx:
                        resname = 'DA5'
                    elif res.name.endswith('G') and res.index == start_idx:
                        resname = 'DG5'
                    elif res.name.endswith('T') and res.index == start_idx:
                        resname = 'DT5'
                    elif res.name.endswith('C') and res.index == start_idx:
                        resname = 'DC5'
                    elif res.name.endswith('A') and res.index == end_idx:
                        resname = 'DA3'
                    elif res.name.endswith('G') and res.index == end_idx:
                        resname = 'DG3'
                    elif res.name.endswith('T') and res.index == end_idx:
                        resname = 'DT3'
                    elif res.name.endswith('C') and res.index == end_idx:
                        resname = 'DC3'
                    elif res.name.startswith('TIP'):
                        resname = 'HOH'
                    else:
                        resname = res.name
                    newRes = newTopology.addResidue(resname, newChain)
                    for atom in res.atoms():
                        newAtom = newTopology.addAtom(atom.name, atom.element, newRes)
            return newTopology
        newTopology = gen_newTop(topology)
        newTopology.loadBondDefinitions('data/g4/residues.xml')
        newTopology.createStandardBonds()

        atoms = list(newTopology.atoms())
        atomnames = [atom.name for atom in newTopology.atoms()]
        o_idx = [i for i in range(len(atomnames)) if atomnames[i] == "O3'"]
        p_idx = [i for i in range(len(atomnames)) if atomnames[i] == "P"]
        for i in range(len(p_idx)):
            newTopology.addBond(atoms[o_idx[i]], atoms[p_idx[i]])
        
        template = dict()

        for res in newTopology.residues():
            template[res] = res.name
        
        return newTopology, template