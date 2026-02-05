import pyagrum as gum

bn = gum.BayesNet('PregnancyTest')
pregnant = bn.add(gum.LabelizedVariable('Pregnant', 'Pregnant', 2))
test = bn.add(gum.LabelizedVariable('Test', 'Test', 2))

bn.addArc(pregnant, test)
bn.cpt(pregnant)[:] = [0.8, 0.2]
bn.cpt(test)[{'Pregnant': 0}] = [0.98, 0.02]
bn.cpt(test)[{'Pregnant': 1}] = [0.01, 0.99]

ie = gum.LazyPropagation(bn)
ie.setEvidence({'Test': 1})
ie.makeInference()
result = ie.posterior(pregnant)[1]

print(f"Posterior P(Pregnant | Positive Test): {result:.4f}")