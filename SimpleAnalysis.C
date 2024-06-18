void SimpleAnalysis(TString FileName, TString outFileName = "out.root")
{
    gROOT->Reset();
    TFile *f = new TFile(FileName.Data(),"READ");
    TFile *outFile = new TFile(outFileName.Data(),"RECREATE");
    outFile->mkdir("momentumFits");
    outFile->mkdir("pTResolutions");
    TTree *events = (TTree*)f->Get("events");
    int NEVENTS = events->GetEntries();
    TTreeReader tr(events);

    TTreeReaderArray<float> gen_px(tr,"GeneratedParticles.momentum.x");
    TTreeReaderArray<float> gen_py(tr,"GeneratedParticles.momentum.y");
    TTreeReaderArray<float> gen_pz(tr,"GeneratedParticles.momentum.z");
    TTreeReaderArray<float> gen_energy(tr,"GeneratedParticles.energy");
    TTreeReaderArray<float> gen_m(tr,"GeneratedParticles.mass");
    TTreeReaderArray<int> gen_pdgid(tr,"GeneratedParticles.PDG");

    TTreeReaderArray<float> rec_px(tr,"ReconstructedChargedParticles.momentum.x");
    TTreeReaderArray<float> rec_py(tr,"ReconstructedChargedParticles.momentum.y");
    TTreeReaderArray<float> rec_pz(tr,"ReconstructedChargedParticles.momentum.z");
    TTreeReaderArray<float> rec_energy(tr,"ReconstructedChargedParticles.energy");
    TTreeReaderArray<float> rec_m(tr,"ReconstructedChargedParticles.mass");
    TTreeReaderArray<int> rec_pdgid(tr,"ReconstructedChargedParticles.PDG");
    
    float _momMin = 80.0;
    float _momMax = 100.0;
    float MomRange = _momMax - _momMin;
    float MomResol = 1.0;
    int nMomBins = (int)(MomRange/MomResol);
    float MomMin[nMomBins], MomMax[nMomBins];
    

    float _pTMin = 0.20;
    float _pTMax = 4.00;
    float pTRange = _pTMax - _pTMin;
    float pTResol = 0.10;
    int npTBins = (int)(pTRange/pTResol);
    float pTMin[npTBins], pTMax[npTBins];
    
    cout << "Number of bins in MOmRage is " << nMomBins << endl;
    for (int i=0; i<nMomBins; i++)
    {
        MomMin[i] = _momMin + i*MomResol;
        MomMax[i] = _momMin + (i+1)*MomResol;
        cout << MomMin[i] << " " << MomMax[i] << endl;
    }
    for (int i=0; i<npTBins; i++)
    {
        pTMin[i] = _pTMin + i*pTResol;
        pTMax[i] = _pTMin + (i+1)*pTResol;
    }
    TH1F* h_mom_res[nMomBins];
    TH1F* h_pT_res[npTBins];
    //TF1 *f_mom_res[nMomBins];
    
    for (int j=0; j<nMomBins; j++)
    {
        h_mom_res[j] = new TH1F(Form("h_mom_res_%d",j),Form("h_mom_res_%d",j),500,-0.2, 0.2);
        cout << h_mom_res[j]->GetName() << endl;
    }
    

    for (int i=0; i<npTBins; i++)
    {
        h_pT_res[i] = new TH1F(Form("h_pT_res_%d",i),Form("h_pT_res_%d",i),500,-0.2, 0.2);
        cout << h_pT_res[i]->GetName() << endl;
    }
    int ev = 0;
    while (tr.Next())
    {
        if (ev+1%100==0) {cout << "Event: " << ev << \
        ", gen_pdg size: " << gen_pdgid.GetSize() << \
        ", recon_pdg size:" << rec_pdgid.GetSize() << endl;}

        if (gen_pdgid.GetSize()!=rec_pdgid.GetSize()) {continue;}
        for (int i=0; i<gen_pdgid.GetSize(); i++)
        {
            if (gen_pdgid[i]!=rec_pdgid[i] || gen_pdgid[i]!=2212) {continue;}
            float gen_mom = sqrt(gen_px[i]*gen_px[i] + gen_py[i]*gen_py[i] + gen_pz[i]*gen_pz[i]);
            float rec_mom = sqrt(rec_px[i]*rec_px[i] + rec_py[i]*rec_py[i] + rec_pz[i]*rec_pz[i]);
            
            for (int j=0; j<nMomBins; j++)
            {
                if (gen_mom>=MomMin[j] && gen_mom<MomMax[j])
                {
                    h_mom_res[j]->Fill((rec_mom-gen_mom)/gen_mom);
                }
            }
            
            float gen_pT = sqrt(gen_px[i]*gen_px[i] + gen_py[i]*gen_py[i]);
            float rec_pT = sqrt(rec_px[i]*rec_px[i] + rec_py[i]*rec_py[i]);
            for (int j=0; j<npTBins; j++)
            {
                if (gen_pT>=pTMin[j] && gen_pT<pTMax[j])
                {
                    h_pT_res[j]->Fill((rec_pT-gen_pT)/gen_pT);
                }
            }
            
        }
        ev++;

    }

    //TCanvas *c = new TCanvas();
    //c->cd();
    //h_mom_res->Draw();
    //c->SaveAs("mom_res.png");
    outFile->cd("momentumFits");
    for (int i=0; i<nMomBins; i++)
    {
        h_mom_res[i]->Write();
    }
    outFile->cd("pTResolutions");
    for (int i=0; i<npTBins; i++)
    {
        h_pT_res[i]->Write();
    }

    
    outFile->Close();

}
