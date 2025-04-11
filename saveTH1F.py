from ROOT import TH1F, TFile
import pickle

f = open('hists_for_ROOT.p', 'rb')
hists = pickle.load(f)
f.close()

# Goal is to write:
# mtt__data_obs  (can't do right now)
# mtt__signal
# mtt__signal__puUp
# mtt__signal__puDown
# mtt__tt_semilep
# mtt__tt_semilep__puUp....

xsec = {'signal':1.0,
        'tt_semilep':831.76*0.438,
        'tt_had':831.76*0.457}

roothists = {}
for sample in hists.keys():
    print(sample)

    genweight = hists[sample]['genWeight']
    Ntot = len(genweight)
    Npos = len(genweight[genweight > 0])
    Nneg = len(genweight[genweight < 0])
    Ngen = sum(genweight)
    print(Ntot,Npos,Nneg,Ngen)

    data = hists[sample]['mtt']
    
    lumiweight = 16400*xsec[sample]*hists[sample]['genWeight']/Ngen 
    nominalweight = lumiweight*hists[sample]['pu_weight'] #...times more?
    puweight_up = lumiweight*hists[sample]['pu_weight_up']
    puweight_dn = lumiweight*hists[sample]['pu_weight_dn']

    if sample == "singlemuon":
        sample = "data_obs"  ## Higgs Combine special name for data histogram

    roothists[sample+'_nominal'] = TH1F("mtt__"+sample,";m_{t#bar{t}} (GeV);events",50,0,3000)
    roothists[sample+'_nominal'].FillN(len(data), data, nominalweight)
    roothists[sample+'_puUp'] = TH1F("mtt__"+sample+"__puUp",";m_{t#bar{t}} (GeV);events",50,0,3000)
    roothists[sample+'_puUp'].FillN(len(data), data, puweight_up)
    roothists[sample+'_puDn'] = TH1F("mtt__"+sample+"__puDn",";m_{t#bar{t}} (GeV);events",50,0,3000)
    roothists[sample+'_puDn'].FillN(len(data), data, puweight_dn)

output = TFile.Open("Zprime_hists.root","recreate")
for ihist in roothists:
    roothists[ihist].Write()

output.Close()

    
#x = np.random.normal(0, 1, 1000)
#hist = ROOT.TH1D("hist", "hist", 50, -3, 3)
#hist.FillN(x.size, x, np.ones(x.size))
