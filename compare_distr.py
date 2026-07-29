#!/usr/bin/env python3
from email.mime import base

import yaml
import ROOT
import glob
import os
import sys
import argparse
import re
import uproot
import numpy as np

from utils import load_and_select_events, analyze_event, check_event_tree

branch_pairs = {
    ("theta", "theta_m2"): {"range": (0, 90), "xlabel": "#theta"},
    ("phi", "phi_m2"): {"range": (-180, 180), "xlabel": "#phi"},
}

def compare_distributions(
    input_file, output_dir, multiplicity_config=None, base_name="params_compare", nbins=50
    
):
    """Compare theta/theta_m2 and phi/phi_m2 for events matching given multiplicities.

    Parameters:
      - input_file: ROOT file containing events and track trees.
      - output_dir: output directory
      - multiplicity_config: multiplicity requirements used for the selection inside this function
    """

    x0_sel, x0_m2_sel, mask1, mask2, tree1, tree2 = check_event_tree(
        input_file, output_dir, multiplicity_config
    )

    norm_hist = False # set to True to normalize histograms to unit area
    cls_separation = True # set to True to separate histograms by number of clusters (2 or 3)

    print(f"\n🔍 --- Comparing angular distributions (x0_mult = {x0_sel}, x0_m2_mult = {x0_m2_sel}) ---") 

    ROOT.gStyle.SetOptStat(0)
    ROOT.gROOT.SetBatch(True)

    c = ROOT.TCanvas("c_compare", "Angular comparison", 1400, 900)

    base = os.path.basename(input_file)
    prefix = base.split("_LVL")[0]

    particle_match = re.search(r"([a-zA-Z])_MAIN", base)
    energy_match = re.search(r"([0-9.]+MeV)", base)

    particle = particle_match.group(1) if particle_match else None
    energy = energy_match.group(1) if energy_match else None

    out_root = os.path.join(output_dir, f"{base_name}_{prefix}_x0_{x0_sel}_x0m2_{x0_m2_sel}.root")
    #fout = ROOT.TFile(out_root, "RECREATE")

    pdf_path = os.path.join(output_dir, f"{base_name}_{prefix}_x0_{x0_sel}_x0m2_{x0_m2_sel}.pdf")
    c.Print(pdf_path + "[") 

    # loop over branches pairs
    for (b1, b2), cgf in branch_pairs.items():
        xmin, xmax = cgf["range"]
        xlabel = cgf.get("xlabel", b1)

        h1 = ROOT.TH1F(f"h_{b1}_cmp", f"{xlabel} distribution - {particle} - {energy};{xlabel} (#circ);Entries", int(nbins), float(xmin), float(xmax))
        h2 = ROOT.TH1F(f"h_{b2}_cmp", f"{xlabel} distribution - {particle} - {energy};{xlabel} (#circ);Entries", int(nbins), float(xmin), float(xmax))
        if cls_separation:
            h1_3c = ROOT.TH1F(f"h_{b1}_cmp_3c", f"{xlabel} distribution - {particle} - {energy};{xlabel} (#circ);Entries", int(nbins), float(xmin), float(xmax))
            h2_3c = ROOT.TH1F(f"h_{b2}_cmp_3c", f"{xlabel} distribution - {particle} - {energy};{xlabel} (#circ);Entries", int(nbins), float(xmin), float(xmax))  
            h1_2c = ROOT.TH1F(f"h_{b1}_cmp_2c", f"{xlabel} distribution - {particle} - {energy};{xlabel} (#circ);Entries", int(nbins), float(xmin), float(xmax))
            h2_2c = ROOT.TH1F(f"h_{b2}_cmp_2c", f"{xlabel} distribution - {particle} - {energy};{xlabel} (#circ);Entries", int(nbins), float(xmin), float(xmax))

        # Read branches
        arr1 = tree1[b1].array(library="np")
        arr2 = tree2[b2].array(library="np")
        cls_1 = tree1["n_cls"].array(library="np")
        cls_2 = tree2["n_cls"].array(library="np")


        # fill with masks
        #print(f"{arr1[mask1]}")
        for i,v in enumerate(arr1[mask1]):
            h1.Fill(float(v))
            if cls_separation:
                if cls_1[i] == 3:
                    h1_3c.Fill(float(v))
                else:
                    h1_2c.Fill(float(v))

        for i,v in enumerate(arr2[mask2]):
            h2.Fill(float(v))
            if cls_separation:
                if cls_2[i] == 3:
                    h2_3c.Fill(float(v))
                else:
                    h2_2c.Fill(float(v))

        if norm_hist:
            h1.Scale(1.0 / h1.Integral(), "width")
            h2.Scale(1.0 / h2.Integral(), "width")
            if cls_separation:
               h1_3c.Scale(1.0 / h1_3c.Integral(), "width") 
               h1_2c.Scale(1.0 / h1_2c.Integral(), "width")
               h2_3c.Scale(1.0 / h2_3c.Integral(), "width")
               h2_2c.Scale(1.0 / h2_2c.Integral(), "width")

        h1.SetLineColor(ROOT.kBlue)
        h2.SetLineColor(ROOT.kRed)
        h1.SetLineWidth(2)
        h2.SetLineWidth(2)
        if cls_separation:
            h1_3c.SetLineColor(ROOT.kBlue+3)
            h2_3c.SetLineColor(ROOT.kRed+3)
            h1_2c.SetLineColor(ROOT.kBlue-7)
            h2_2c.SetLineColor(ROOT.kRed-7)
            h1_3c.SetLineWidth(1)
            h2_3c.SetLineWidth(1)
            h1_2c.SetLineWidth(1)
            h2_2c.SetLineWidth(1)
            h1_3c.SetLineStyle(2)
            h2_3c.SetLineStyle(2)
            h1_2c.SetLineStyle(3)
            h2_2c.SetLineStyle(3)

        ymax = max(h1.GetMaximum(), h2.GetMaximum()) * 1.2
        h1.SetMaximum(ymax)
        h2.SetMaximum(ymax)
        h1.SetMinimum(0.0)
        h2.SetMinimum(0.0)

        # Draw
        h1.Draw("HIST")
        h2.Draw("HIST SAME")
        if cls_separation:
            h1_3c.Draw("HIST SAME")
            h2_3c.Draw("HIST SAME")
            h1_2c.Draw("HIST SAME")
            h2_2c.Draw("HIST SAME")

        if xlabel == "#theta":
            leg = ROOT.TLegend(0.55, 0.75, 0.93, 0.93)
        else:
            if particle != "c":
                leg = ROOT.TLegend(0.20, 0.73, 0.58, 0.93)
            else:
                leg = ROOT.TLegend(0.55, 0.75, 0.93, 0.93)
        leg.AddEntry(h1, f"Hough - Entries: {int(h1.GetEntries())}", "l")
        leg.AddEntry(h2, f"Comb. - Entries: {int(h2.GetEntries())}", "l")
        if cls_separation:
            leg.AddEntry(h1_3c, f"Hough 3-cls - Entries: {int(h1_3c.GetEntries())}", "l")
            leg.AddEntry(h2_3c, f"Comb. 3-cls - Entries: {int(h2_3c.GetEntries())}", "l")
            leg.AddEntry(h1_2c, f"Hough 2-cls - Entries: {int(h1_2c.GetEntries())}", "l")
            leg.AddEntry(h2_2c, f"Comb. 2-cls - Entries: {int(h2_2c.GetEntries())}", "l")
        #leg.AddEntry(h1, f"Entries: {h1.GetEntries()}", "")
        #leg.AddEntry(h2, f"Entries: {h2.GetEntries()}", "")
        leg.SetTextSize(0.03)
        leg.Draw()

        c.Print(pdf_path)

        #fout.cd()
        #h1.Write()
        #h2.Write()

    c.Print(pdf_path + "]")
    #fout.Close()

    print(f"\n✅ Saved comparison plots to {pdf_path}")
    print(f"✅ Saved histograms to {out_root}")


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(
        description="Compare track multiplicity histograms across ROOT files. To be executed after check_extra_tracks.py with all the cuts applied."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input root file with distributions to compare. Root file *selected_m1_vs_m2.root generated by check_extra_tracks.py.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory where results will be stored",
    )

    args = parser.parse_args()

    input_file = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)

    # read multiplicity cuts from config
    with open("config/config_check_hough.yaml", "r") as f_cfg:
        config = yaml.safe_load(f_cfg)
    multiplicity_config = config["multiplicity_config"]
    
    compare_distributions(input_file, output_dir, multiplicity_config)