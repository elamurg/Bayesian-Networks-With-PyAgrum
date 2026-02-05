import pyagrum as gum
import pyagrum.lib.notebook as gnb

bn = gum.BayesNet('CarElectricalSystem')
battery = bn.add(gum.LabelizedVariable('Battery', 'Battery', 2))
gas = bn.add(gum.LabelizedVariable('Gas', 'Gas', 2))
ignition = bn.add(gum.LabelizedVariable('Ignition', 'Ignition', 2))
starter = bn.add(gum.LabelizedVariable('EngineStarts', 'Engine Starts'))
radio = bn.add(gum.LabelizedVariable('Radio', 'Radio', 2))
lights = bn.add(gum.LabelizedVariable('Lights', 'Lights', 2))
bn.addArc(battery, ignition)
bn.addArc(battery, radio)
bn.addArc(battery, lights)
bn.addArc(ignition, starter)
bn.addArc(gas, starter)

#node probability tables (0 = False, 1 = True)
bn.cpt(battery)[:] = [0.1, 0.9]
bn.cpt(gas)[:] = [0.25, 0.75]
bn.cpt(ignition)[{'Battery': 0}] = [1, 0]
bn.cpt(ignition)[{'Battery': 1}] = [0.10, 0.90]
bn.cpt(radio)[{'Battery': 0}] = [1, 0]
bn.cpt(radio)[{'Battery': 1}] = [0.05, 0.95]
bn.cpt(lights)[{'Battery': 0}] = [1, 0]
bn.cpt(lights)[{'Battery': 1}] = [0.05, 0.95]
bn.cpt(starter)[{'Ignition': 0, 'Gas': 0}] = [1, 0]
bn.cpt(starter)[{'Ignition': 0, 'Gas': 1}] = [1, 0]
bn.cpt(starter)[{'Ignition': 1, 'Gas': 0}] = [1, 0]
bn.cpt(starter)[{'Ignition': 1, 'Gas': 1}] = [0.05, 0.95]

#inference
ie = gum.LazyPropagation(bn)
ie.setEvidence({'Lights': 1})
ie.makeInference()
prob_a = ie.posterior(starter)[1]
print(f"3a. P(EngineStarts | Lights=True): {prob_a:.4f}")

ie.setEvidence({'Lights': 1, 'Gas': 1})
ie.makeInference()
print(f"3b. P(EngineStarts | Lights=True, Gas=True): {ie.posterior(starter)[1]:.4f}")