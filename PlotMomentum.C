

/**
 * Plots the momentum and pT resolutions for protons in a given file.
 *
 * @param FileName The name of the input file.
 * @param outFileSig The output file signature (default is "out").
 */
void PlotMomentum(TString FileName, TString outFileSig = "out")
{
    gStyle->SetTitleSize(0.08);
    gStyle->SetOptFit(11);

    TFile *f = new TFile(FileName.Data(),"READ");
    TFile *outFile = new TFile(outFileSig + ".root", "RECREATE");
    outFile->mkdir("momentumFits");
    outFile->mkdir("pTResolutions");
    outFile->cd("momentumFits");

    float _momMin = 80.0;
    float _momMax = 100.0;
    float MomRange = _momMax - _momMin;
    float MomResol = 1.0;
    const int nMomBins = (int)(MomRange/MomResol);
    float _pTMin = 0.20;
    float _pTMax = 4.00;
    float pTResol = 0.10;
    const int npTBins = (int)((_pTMax - _pTMin)/pTResol);
    float MomMin[nMomBins], MomMax[nMomBins];
    float pTMin[npTBins], pTMax[npTBins];
    for (int i=0; i<nMomBins; i++)
    {
        MomMin[i] = _momMin + i*MomResol;
        MomMax[i] = _momMin + (i+1)*MomResol;
    }
    for (int i=0; i<npTBins; i++)
    {
        pTMin[i] = _pTMin + i*pTResol;
        pTMax[i] = _pTMin + (i+1)*pTResol;
    }
    TH1F *h_mom_res[nMomBins];
    TH1F *h_pT_res[npTBins];
    TCanvas *canvasMom[nMomBins];
    TCanvas *canvaspT[npTBins];
    TF1 *f_mom_res[nMomBins];
    TF1 *f_pT_res[npTBins];
    float dev = 5;
    float mean = 0.;
    float sigma = 0.2;
    float resols[nMomBins];
    float resols_err[nMomBins];
    float A1;
    float A2;
    float sigma1;
    float sigma2;
    float pT_dev = 5;
    float pT_mean = 0.;
    float pT_sigma = 0.2;
    float pT_resols[npTBins];
    float pT_resols_err[npTBins];
    float pT_A1;
    float pT_A2;
    float pT_sigma1;
    float pT_sigma2;
    for (int i=0; i<nMomBins; i++)
    {
        h_mom_res[i] = (TH1F*)f->Get(Form("momentumFits/h_mom_res_%d",i));
        h_mom_res[i]->SetTitle(Form("protons %.2f < p < %.2f [GeV];#vec{p^{gen}};#frac{#Delta p}{p}", MomMin[i], MomMax[i]));
        h_mom_res[i]->Rebin(4);
        canvasMom[i] = new TCanvas(Form("canvasMom_%d",i), Form("canvasMom_%d",i));
        sigma = h_mom_res[i]->GetRMS();
        
        f_mom_res[i] = new TF1(Form("f_mom_res_%d",i),"gaus+gaus(3)",-dev*sigma,dev*sigma);
        f_mom_res[i]->SetParameters(h_mom_res[i]->GetMaximum()*0.8,mean,0.1*sigma, h_mom_res[i]->GetMaximum()*0.1,mean,0.5*sigma);
        h_mom_res[i]->Fit(Form("f_mom_res_%d",i),"R");

        A1 = f_mom_res[i]->GetParameter(0);
        A2 = f_mom_res[i]->GetParameter(3);
        sigma1 = f_mom_res[i]->GetParameter(2);
        sigma2 = f_mom_res[i]->GetParameter(5);
        resols[i] = (A1*sigma1 + A2*sigma2)/(A1 + A2);// (A1*sigma1 + A2*sigma2) / (A1 + A2)
        resols_err[i] = (A1/(A1+A2))*f_mom_res[i]->GetParError(2) + (A2/(A1+A2))*f_mom_res[i]->GetParError(5);
        resols_err[i]*=(f_mom_res[i]->GetChisquare()/f_mom_res[i]->GetNDF());

    }
    for (int i=0; i<npTBins; i++)
    {
        h_pT_res[i] = (TH1F*)f->Get(Form("pTResolutions/h_pT_res_%d",i));
        h_pT_res[i]->Rebin(4);
        h_pT_res[i]->SetTitle(Form("protons %.2f < p_{T} < %.2f [GeV];#vec{p_{T}^{gen}};#frac{#Delta p_{T}}{p_{T}}", pTMin[i], pTMax[i]));
        canvaspT[i] = new TCanvas(Form("canvaspT_%d",i), Form("canvaspT_%d",i));
        pT_sigma = h_pT_res[i]->GetRMS();

        f_pT_res[i] = new TF1(Form("f_pT_res_%d",i),"gaus+gaus(3)",-pT_dev*pT_sigma,pT_dev*pT_sigma);
        f_pT_res[i]->SetParameters(h_pT_res[i]->GetMaximum()*0.8,pT_mean,0.1*pT_sigma, h_pT_res[i]->GetMaximum()*0.1,pT_mean,0.5*pT_sigma);
        h_pT_res[i]->Fit(Form("f_pT_res_%d",i),"R");

        pT_A1 = f_pT_res[i]->GetParameter(0);
        pT_A2 = f_pT_res[i]->GetParameter(3);
        pT_sigma1 = f_pT_res[i]->GetParameter(2);
        pT_sigma2 = f_pT_res[i]->GetParameter(5);
        pT_resols[i] = h_pT_res[i]->GetRMS();//(pT_A1*pT_sigma1 + pT_A2*pT_sigma2)/(pT_A1 + pT_A2);// (A1*sigma1 + A2*sigma2) / (A1 + A2)
        pT_resols_err[i] = (pT_A1/(pT_A1+pT_A2))*f_pT_res[i]->GetParError(2) + (pT_A2/(pT_A1+pT_A2))*f_pT_res[i]->GetParError(5);
        pT_resols_err[i]*=(f_pT_res[i]->GetChisquare()/f_pT_res[i]->GetNDF());  

    }
    for (int i = 0; i<nMomBins; i++)
    {
        canvasMom[i]->cd();
        h_mom_res[i]->Draw();
        canvasMom[i]->Write();
        if (i==0) canvasMom[i]->Print(outFileSig + "_mom_res.pdf(");
        else if (i==nMomBins-1) canvasMom[i]->Print(outFileSig + "_mom_res.pdf)");
        else canvasMom[i]->Print(outFileSig + "_mom_res.pdf");
    }
    outFile->cd("pTResolutions");
    for (int i = 0; i<npTBins; i++)
    {
        canvaspT[i]->cd();
        h_pT_res[i]->Draw();
        canvaspT[i]->Write();
        if (i==0) canvaspT[i]->Print(outFileSig + "_pT_res.pdf(");
        else if (i==npTBins-1) canvaspT[i]->Print(outFileSig + "_pT_res.pdf)");
        else canvaspT[i]->Print(outFileSig + "_pT_res.pdf");
    }
    outFile->cd();
    float centerMomBins[nMomBins];
    float centerMomBinErrs[nMomBins];
    for (int i=0; i<nMomBins; i++)
    {
        centerMomBins[i] = (MomMin[i] + MomMax[i])/2;
        centerMomBinErrs[i] = (MomMax[i] - MomMin[i])/2;
    }
    float centerpTBins[npTBins];
    float centerpTBinErrs[npTBins];
    for (int i=0; i<npTBins; i++)
    {
        centerpTBins[i] = (pTMin[i] + pTMax[i])/2;
        centerpTBinErrs[i] = (pTMax[i] - pTMin[i])/2;
    }
    TGraphErrors *gr_res = new TGraphErrors(nMomBins, centerMomBins, resols, centerMomBinErrs, resols_err);
    gr_res->SetName(outFileSig + "_res");
    gr_res->SetTitle(";#vec{p^{gen}};#frac{#sigma_{p}}{p}");
    gr_res->GetXaxis()->SetTitleSize(0.075);
    gr_res->GetYaxis()->SetTitleSize(0.075);
    gr_res->GetXaxis()->SetLabelSize(0.06);
    gr_res->GetYaxis()->SetLabelSize(0.06);
    gr_res->SetMaximum(0.1);
    gr_res->SetMinimum(0.0);
    TCanvas *c_res = new TCanvas(outFileSig + "_canvas_res", outFileSig + "_canvas_res", 1200, 800);
    c_res->SetLeftMargin(0.2);
    c_res->SetBottomMargin(0.2);
    gr_res->SetMarkerStyle(20);
    gr_res->SetMarkerColor(kRed);
    gr_res->Draw("APE");
    c_res->Print(outFileSig + "_res.pdf");
    gr_res->Write();
    c_res->Write();

    TGraphErrors *gr_res_pT = new TGraphErrors(npTBins, centerpTBins, pT_resols, centerpTBinErrs, pT_resols_err);
    gr_res_pT->SetName(outFileSig + "_res_pT");
    gr_res_pT->SetTitle(";#vec{p_{T}^{gen}};#frac{#sigma_{p_{T}}}{p_{T}}");
    gr_res_pT->GetXaxis()->SetTitleSize(0.075);
    gr_res_pT->GetYaxis()->SetTitleSize(0.075);
    gr_res_pT->GetXaxis()->SetLabelSize(0.06);
    gr_res_pT->GetYaxis()->SetLabelSize(0.06);
    gr_res_pT->SetMaximum(0.1);
    gr_res_pT->SetMinimum(0.0);
    TCanvas *c_res_pT = new TCanvas(outFileSig + "_canvas_res_pT", outFileSig + "_canvas_res_pT", 1200, 800);
    c_res_pT->SetLeftMargin(0.2);
    c_res_pT->SetBottomMargin(0.2);
    gr_res_pT->SetMarkerStyle(20);
    gr_res_pT->SetMarkerColor(kRed);
    gr_res_pT->Draw("APE");
    c_res_pT->Print(outFileSig + "_res_pT.pdf");
    gr_res_pT->Write();
    c_res_pT->Write();


    outFile->Close();

}
