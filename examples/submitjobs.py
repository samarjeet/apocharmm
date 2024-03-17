import subprocess

timeSteps = [2.0] # [1.0]
frictionCoefficients = [0.0, 2.0, 5.0, 12.0, 20.0]

pressurePistonMassFactors = [5.0, 10.0, 20.0]
temperaturePistonMassFactors = [10.0, 15., 20.0]

for timeStep in timeSteps:
    for frictionCoefficient in frictionCoefficients:
        for pressurePistonMassFactor in pressurePistonMassFactors:
            for temperaturePistonMassFactor in temperaturePistonMassFactors:
                bashCommand = f'sbatch run.slurm {timeStep} {frictionCoefficient} {pressurePistonMassFactor} {temperaturePistonMassFactor}'
                process = subprocess.Popen(
                    bashCommand.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
                print(output)
