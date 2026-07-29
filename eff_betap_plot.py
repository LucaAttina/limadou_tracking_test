import argparse
import os
import numpy as np
import ROOT
import glob
import json


def eff_fake_plots(input_dir, output_dir):

    files = sorted(glob.glob(os.path.join(input_dir, "*.json")))

    if not files:
        raise RuntimeError(f"No .json files found in {input_dir}")

    print("\nFiles to be processed:")
    print("-" * 60)
    for f in files:
        print(os.path.basename(f))
    print("-" * 60 + "\n")

    ROOT.gStyle.SetOptStat(0)
    #ROOT.gStyle.SetTitleSize(0.06, "")

    variables = ["efficiency", "fake_rate"]
    methods = ["method1", "method2"]
    
    # Mappa per i nomi dei metodi
    method_labels = {
        "method1": "m1",
        "method2": "m2"
    }
    
    # Mappa per i nomi delle variabili
    var_labels = {
        "efficiency": "Efficiency",
        "fake_rate": "Fake Rate"
    }

    colors = {
        "e": ROOT.kBlack,
        "p": ROOT.kRed,
        "c": ROOT.kGreen
    }
    
    markers = {
        "method1": 20,
        "method2": 26
    }
    
    line_styles = {
        "method1": 1,
        "method2": 2
    }

    # =========================================================
    # Canvas
    # =========================================================

    c = ROOT.TCanvas("c", "Eff_Fake", 2500, 1100)

    # 2x2 grid: efficiency (left), fake_rate (right)
    # top: main plots, bottom: ratio plots
    pad1 = ROOT.TPad("pad1", "", 0.00, 0.30, 0.5, 1.00)  # efficiency
    pad2 = ROOT.TPad("pad2", "", 0.50, 0.30, 1.00, 1.00)  # fake_rate
    
    pad3 = ROOT.TPad("pad3", "", 0.00, 0.00, 0.5, 0.30)   # efficiency ratio
    pad4 = ROOT.TPad("pad4", "", 0.50, 0.00, 1.00, 0.30)   # fake_rate ratio

    pads = [pad1, pad2]
    ratio_pads = [pad3, pad4]

    for i, (pad, r_pad) in enumerate(zip(pads, ratio_pads)):
        pad.SetLogx()
        r_pad.SetLogx()

        if i == 1:
            r_pad.SetLogy() 
            pad.SetLogy() 
        
        pad.SetTopMargin(0.08)
        pad.SetBottomMargin(0)
        r_pad.SetTopMargin(0)
        r_pad.SetBottomMargin(0.4)
        
        if i == 0:
            pad.SetLeftMargin(0.12)
            pad.SetRightMargin(0.07)
            r_pad.SetLeftMargin(0.12)
            r_pad.SetRightMargin(0.07)
        else:
            pad.SetLeftMargin(0.09)
            pad.SetRightMargin(0.12)
            r_pad.SetLeftMargin(0.09)
            r_pad.SetRightMargin(0.12)
        
        pad.Draw()
        r_pad.Draw()

    # =========================================================
    # Legenda unica
    # =========================================================

    legend = ROOT.TLegend(0.80, 0.40, 0.99, 0.53)
    legend.SetBorderSize(1)
    legend.SetFillStyle(1001)
    legend.SetTextSize(0.037)
    legend.SetNColumns(2)

    # =========================================================
    # Liste per mantenere i riferimenti ai grafici
    # =========================================================
    
    all_graphs = []  # Per i grafici principali
    all_bands = []   # Per le bande di errore
    all_ratio_graphs = []  # Per i grafici ratio
    all_ratio_bands = []   # Per le bande di errore dei ratio

    # =========================================================
    # Processa i file e trova range globale
    # =========================================================

    global_min = {}
    global_max = {}
    
    for var in variables:
        global_min[var] = 1e9
        global_max[var] = -1e9

    datasets = []

    for fname in files:
        print(f"Processing {os.path.basename(fname)}")
        
        with open(fname) as jf:
            data = json.load(jf)
        
        particle = data["particle"]
        betap_vals = np.array(data["betap"], dtype=float)
        
        dataset = {
            "particle": particle,
            "betap": betap_vals,
            "methods": {}
        }
        
        for meth in methods:
            dataset["methods"][meth] = {}
            for var in variables:
                y = np.array(data[meth][var], dtype=float)
                yerr = np.array(data[meth][var + "_err"], dtype=float)
                
                # Ordina per betap
                points = sorted(zip(betap_vals, y, yerr), key=lambda t: t[0])
                betap_sorted, y_sorted, yerr_sorted = map(np.array, zip(*points))
                
                dataset["methods"][meth][var] = {
                    "betap": betap_sorted,
                    "value": y_sorted,
                    "error": yerr_sorted
                }
                
                # Aggiorna range globale
                ymin = np.min(y_sorted - yerr_sorted)
                ymax = np.max(y_sorted + yerr_sorted)
                if ymin > 0:
                    global_min[var] = min(global_min[var], ymin)
                global_max[var] = max(global_max[var], ymax)
        
        datasets.append(dataset)

    # Aggiungi margini ai range
    for var in variables:
        if var == "efficiency":
            # Efficiency: range 0-1 con un po' di margine
            global_min[var] = max(0, global_min[var] * 0.95)
            global_max[var] = min(1.0, global_max[var] * 1.05)
        elif var == "fake_rate":
            # Fake rate: scala log, margine maggiore
            if global_min[var] > 0:
                global_min[var] *= 0.5  # più margine per il fake rate
            else:
                global_min[var] = 1e-6
            global_max[var] *= 2.0

        # Print dei range per controllo
    print("\n" + "="*60)
    print("GLOBAL RANGES:")
    for var in variables:
        print(f"  {var}: [{global_min[var]:.6f}, {global_max[var]:.6f}]")
    print("="*60 + "\n")


    # =========================================================
    # Disegno
    # =========================================================

    for ivar, var in enumerate(variables):
        pad = pads[ivar]
        pad.cd()
        
        first_graph = True
        
        for dataset in datasets:
            particle = dataset["particle"]
            
            for meth in methods:
                data = dataset["methods"][meth][var]
                bp = data["betap"]
                y = data["value"]
                yerr = data["error"]
                n = len(bp)

                # ============ PRINT PER VERIFICA ============
                print(f"\n{'='*60}")
                print(f"MAIN PLOT: {var} - {particle} - {meth}")
                print(f"  Number of points: {n}")
                print(f"  betap range: [{bp[0]:.2f}, {bp[-1]:.2f}]")
                print(f"  y range: [{np.min(y):.6f}, {np.max(y):.6f}]")
                print(f"  y mean: {np.mean(y):.6f}")
                print(f"  yerr range: [{np.min(yerr):.6f}, {np.max(yerr):.6f}]")
                print(f"  First 3 points:")
                for j in range(min(3, n)):
                    print(f"    betap={bp[j]:.2f}, y={y[j]:.6f}, yerr={yerr[j]:.6f}")
                print(f"{'='*60}")
                # =============================================
                
                # Crea graph
                graph = ROOT.TGraphErrors(n)
                band = ROOT.TGraphErrors(n)
                graph.SetTitle("")
                band.SetTitle("")
                
                for j in range(n):
                    graph.SetPoint(j, bp[j], y[j])
                    graph.SetPointError(j, 0, yerr[j])
                    band.SetPoint(j, bp[j], y[j])
                    band.SetPointError(j, 0, yerr[j])
                
                # Style
                color = colors[particle]
                marker_style = markers[meth]
                line_style = line_styles[meth]
                
                graph.SetLineColor(color)
                graph.SetMarkerColor(color)
                graph.SetLineWidth(1)
                graph.SetLineStyle(1)
                graph.SetMarkerStyle(marker_style)
                graph.SetMarkerSize(1.8)
                
                band.SetFillColorAlpha(color, 0.25)
                band.SetLineColor(color)
                band.SetLineWidth(0)
                
                # Salva nei riferimenti
                all_graphs.append(graph)
                all_bands.append(band)
                
                # Draw
                if first_graph:
                    graph.Draw("AP")
                    if ivar == 0:
                        graph.GetYaxis().SetRangeUser(0.1, 1.1)
                    else:
                        graph.GetYaxis().SetRangeUser(3e-7, 0.15)

                    graph.GetXaxis().SetLimits(5.0, 9500)
                    
                    if ivar != 0:
                        graph.GetYaxis().SetLabelSize(0)
                        graph.GetYaxis().SetTitleSize(0)
                    
                    graph.GetXaxis().SetLabelSize(0.053)
                    graph.GetYaxis().SetLabelSize(0.053)
                    graph.GetXaxis().SetTitleSize(0.05)
                    graph.GetYaxis().SetTitleSize(0.05)
                    graph.GetYaxis().SetTitleOffset(1.4)
                    
                    first_graph = False
                else:
                    graph.Draw("P SAME")
                
                band.Draw("3 SAME")
                
                # Legenda
                if ivar == 0:
                    if method_labels[meth] == "m1":
                        label = f"{particle} Hough"
                    else:
                        label = f"{particle} Comb."
                    #label = f"{particle} {method_labels[meth]}"
                    legend.AddEntry(graph, label, "lp")

        # Titolo del pannello
        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.045)
        latex.SetTextFont(42)
        
        #x_pos = 0.25 if ivar == 0 else 0.75
        #latex.DrawLatex(x_pos - 0.1, 0.93, var_labels[var])

    # =========================================================
    # Ratio plots
    # =========================================================

    for ivar, var in enumerate(variables):
        pad = ratio_pads[ivar]
        pad.cd()
        
        first_graph = True
        
        for dataset in datasets:
            particle = dataset["particle"]
            
            data1 = dataset["methods"]["method1"][var]
            data2 = dataset["methods"]["method2"][var]
            
            bp = data1["betap"]
            y1 = data1["value"]
            yerr1 = data1["error"]
            y2 = data2["value"]
            yerr2 = data2["error"]
            
            n = len(bp)

            # Calcola ratio per il print
            ratio_values = y2 / y1
            
            # ============ PRINT PER VERIFICA RATIO ============
            print(f"\n{'='*60}")
            print(f"RATIO PLOT: {var} - {particle}")
            print(f"  Number of points: {n}")
            print(f"  betap range: [{bp[0]:.2f}, {bp[-1]:.2f}]")
            print(f"  y1 (method1) range: [{np.min(y1):.6f}, {np.max(y1):.6f}]")
            print(f"  y2 (method2) range: [{np.min(y2):.6f}, {np.max(y2):.6f}]")
            print(f"  ratio (y2/y1) range: [{np.min(ratio_values):.4f}, {np.max(ratio_values):.4f}]")
            print(f"  ratio mean: {np.mean(ratio_values):.4f}")
            print(f"  First 3 points:")
            for j in range(min(3, n)):
                print(f"    betap={bp[j]:.2f}, y1={y1[j]:.6f}, y2={y2[j]:.6f}, ratio={ratio_values[j]:.4f}")
            print(f"{'='*60}")
            # ===================================================
            
            graph = ROOT.TGraphErrors(n)
            band = ROOT.TGraphErrors(n)
            graph.SetTitle("")
            band.SetTitle("")
            
            for j in range(n):
                ratio = y2[j] / y1[j]
                ratio_err = ratio * np.sqrt((yerr2[j] / y2[j])**2 + (yerr1[j] / y1[j])**2)
                
                graph.SetPoint(j, bp[j], ratio)
                graph.SetPointError(j, 0, ratio_err)
                band.SetPoint(j, bp[j], ratio)
                band.SetPointError(j, 0, ratio_err)
            
            # Style
            color = colors[particle]
            marker_style = markers["method2"]
            
            graph.SetLineColor(color)
            graph.SetMarkerColor(color)
            graph.SetLineWidth(1)
            graph.SetLineStyle(1)
            graph.SetMarkerStyle(marker_style)
            graph.SetMarkerSize(1.8)
            
            band.SetFillColorAlpha(color, 0.25)
            band.SetLineColor(color)
            band.SetLineWidth(0)
            
            # Salva nei riferimenti
            all_ratio_graphs.append(graph)
            all_ratio_bands.append(band)
            
            # Draw
            if first_graph:
                graph.Draw("AP")
                if ivar == 0:
                    graph.GetYaxis().SetRangeUser(0, 2.0)
                else:
                    graph.GetYaxis().SetRangeUser(0.1, 200)
                graph.GetXaxis().SetLimits(5.0, 9500)

                if var == "fake_rate":
                    pad.SetLogy()
                
                if ivar != 0:
                    graph.GetYaxis().SetLabelSize(0)
                    graph.GetYaxis().SetTitleSize(0)
                
                graph.GetXaxis().SetLabelSize(0.12)
                graph.GetYaxis().SetLabelSize(0.12)
                graph.GetXaxis().SetTitleSize(0.05)
                graph.GetYaxis().SetTitleSize(0.05)
                graph.GetYaxis().SetTitleOffset(0.8)
                
                # Linea a y=1
                line = ROOT.TLine()
                line.SetLineStyle(2)
                line.SetLineColor(ROOT.kGray + 1)
                line.SetLineWidth(2)
                line.DrawLine(5.0, 1.0, 9500, 1.0)
                
                first_graph = False
            else:
                graph.Draw("P SAME")
            
            band.Draw("3 SAME")

    # =========================================================
    # Label globali
    # =========================================================

    c.cd()
    c.Update()

    xlabel = ROOT.TLatex()
    xlabel.SetNDC()
    xlabel.SetTextSize(0.042)
    xlabel.SetTextFont(42)
    xlabel.DrawLatex(0.22, 0.03, "#betap [MeV]")
    xlabel.DrawLatex(0.70, 0.03, "#betap [MeV]")

    ylabel1 = ROOT.TLatex()
    ylabel1.SetNDC()
    ylabel1.SetTextAngle(90)
    ylabel1.SetTextSize(0.042)
    ylabel1.SetTextFont(42)
    ylabel1.DrawLatex(0.020, 0.60, "Efficiency")
    ylabel1.DrawLatex(0.50, 0.60, "Fake-Track Rate")

    ylabel2 = ROOT.TLatex()
    ylabel2.SetNDC()
    ylabel2.SetTextAngle(90)
    ylabel2.SetTextSize(0.042)
    ylabel2.SetTextFont(42)
    ylabel2.DrawLatex(0.020, 0.09, "Comb. / Hough")
    ylabel2.DrawLatex(0.50, 0.09, "Comb. / Hough")

    legend.Draw()

    c.Update()

    outname = os.path.join(output_dir, "eff_fake_plots.pdf")
    c.SaveAs(outname)

    print(f"\nSaved plot to: {outname}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Efficiency and fake rate plots"
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing json files"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    eff_fake_plots(
        args.input_dir,
        args.output_dir
    )