void CompareMomentum(TString outFileName = "MomentumResolution")
{
    TLegend *leg = new TLegend(0.6, 0.6, 0.9, 0.9);
    TString defaultFileName = "April01_DefaultConfigurationResult.root";
    TString OtherFileName[2] = {"April02_Delta10cmConfigurationResult.root", "April01_Delta20cmConfigurationResult.root"};
    TString HistNames[2] = {"April02_Delta10cmConfigurationResult_res_pT", "April01_Delta20cmConfigurationResult_res_pT"};
    TString Labels[3] = {"Default Configuration", "Delta 10 cm Configuration", "Delta 20 cm Configuration"};

    TFile *f = TFile::Open(defaultFileName.Data());
    TGraphErrors *g_mom_res = (TGraphErrors*)f->Get("April01_DefaultConfigurationResult_res_pT");
    g_mom_res->SetMarkerSize(2);
    g_mom_res->SetMarkerColor(kBlack);
    g_mom_res->SetMarkerStyle(20);
    leg->AddEntry(g_mom_res, "Default Configuration", "ep");

    TFile *otherFiles[2];
    TGraphErrors *g_mom_res_other[2];
    int Colors[2] = {kRed, kBlue};
    for (int i=0; i<2; i++)
    {
        otherFiles[i] = TFile::Open(OtherFileName[i].Data());
        g_mom_res_other[i] = (TGraphErrors*)otherFiles[i]->Get(HistNames[i].Data());
        g_mom_res_other[i]->SetMarkerSize(2);
        g_mom_res_other[i]->SetMarkerColor(Colors[i]);
        g_mom_res_other[i]->SetMarkerStyle(20);
        leg->AddEntry(g_mom_res_other[i], Labels[i+1].Data(), "ep");
    }

    TCanvas *c1 = new TCanvas("c1", "c1", 1200, 800);
    c1->SetLeftMargin(0.2);
    c1->SetBottomMargin(0.2);
    c1->cd();
    g_mom_res->Draw("AP");
    for (int i=0; i<2; i++)
    {
        g_mom_res_other[i]->Draw("SAME P");
    }
    leg->Draw("SAME");
    c1->Print(outFileName + "_res_pT.pdf");

    TFile *outFile = new TFile(outFileName + ".root", "RECREATE");
    g_mom_res->Write();
    for (int i=0; i<2; i++)
    {
        g_mom_res_other[i]->Write();
    }
    c1->Write();
    outFile->Close();


}