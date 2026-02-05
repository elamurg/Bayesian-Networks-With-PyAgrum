import pyagrum as gum
import pyagrum.lib.notebook as gnb

bn = gum.BayesNet('Asia')
s = bn.add(gum.LabelizedVariable('Smoking', 'Smoking', ['No', 'Yes'] ))
c = bn.add(gum.LabelizedVariable('Lung_Cancer', 'Lung Cancer', ['No', 'Yes']))
b = bn.add(gum.LabelizedVariable('Bronchitis', 'Bronchitis', ['No', 'Yes']))
d = bn.add(gum.LabelizedVariable('Dyspnoea', 'Dyspnoea', ['No', 'Yes']))

bn.addArc(s, c)
bn.addArc(s, b)
bn.addArc(c, d)
bn.addArc(b, d)

gnb.showBN(bn)