python apply-preselection.py --input "root://eospublic.cern.ch//eos/opendata/cms/Run2016H/SingleMuon/NANOAOD/UL2016_MiniAODv2_NanoAODv9-v1/*/*.root"\
 --output /eos/user/l/ligerlac/z-prime-ttbar-data/data.root

python apply-preselection.py --input "root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/ZprimeToTT_M2000_W20_TuneCP2_PSweights_13TeV-madgraph-pythiaMLM-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1/*/*.root"\
 --output /eos/user/l/ligerlac/z-prime-ttbar-data/z_prime.root --is-mc

python apply-preselection.py --input "root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1/*/*.root"\
 --output /eos/user/l/ligerlac/z-prime-ttbar-data/tt_semilep.root --is-mc

python apply-preselection.py --input "root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1/*/*.root"\
 --output /eos/user/l/ligerlac/z-prime-ttbar-data/tt_had.root --is-mc

python apply-preselection.py --input "root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1/*/*.root"\
 --output /eos/user/l/ligerlac/z-prime-ttbar-data/tt_lep.root --is-mc

python apply-preselection.py --input "root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16NanoAODv9/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_v17-v1/*/*.root"\
 --output /eos/user/l/ligerlac/z-prime-ttbar-data/w_jets.root --is-mc
