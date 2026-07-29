import argparse
import os
import numpy as np
import ROOT
import glob
import json


def res_plots(input_dir, output_dir):

    files = sorted(glob.glob(os.path.join(input_dir, "*.json")))

    if not files:
        raise RuntimeError(f"No .json files found in {input_dir}")

    print("\nFiles to be processed:")
    print("-" * 60)
    for f in files:
        print(os.path.basename(f))
    print("-" * 60 + "\n")

    ROOT.gStyle.SetOptStat(0)
    #ROOT.gStyle.SetTitleFont(42, "")          
    ROOT.gStyle.SetTitleSize(0.06, "")  

    layers = ["l1", "l2", "l3"]
    methods = ["m1", "m2"]

    colors = {
        "electron": ROOT.kBlack,
        "proton": ROOT.kRed,
        "carbon": ROOT.kGreen+3
    }

    # =========================================================
    # Canvas
    # =========================================================

    c = ROOT.TCanvas("c", "Residuals", 3000, 1100)

    # Pad attaccati
    pad1 = ROOT.TPad("pad1", "", 0.00, 0.30, 0.35, 1.00)
    pad2 = ROOT.TPad("pad2", "", 0.35, 0.30, 0.666, 1.00)
    pad3 = ROOT.TPad("pad3", "", 0.666, 0.30, 1.00, 1.00)

    pad4 = ROOT.TPad("pad4", "", 0.00, 0.00, 0.35, 0.30)
    pad5 = ROOT.TPad("pad5", "", 0.35, 0.00, 0.666, 0.30)
    pad6 = ROOT.TPad("pad6", "", 0.666, 0.00, 1.00, 0.30)

    pads = [pad1, pad2, pad3]
    ratio_pads = [pad4, pad5, pad6]

    for i, (pad, r_pad) in enumerate(zip(pads, ratio_pads)):

        pad.SetLogx()
        pad.SetLogy()
        r_pad.SetLogx()

        pad.SetTopMargin(0.08)
        pad.SetBottomMargin(0)
        r_pad.SetTopMargin(0)
        r_pad.SetBottomMargin(0.4)

        # margini uguali
        #pad.SetLeftMargin(0.0)
        if i == 0:
            pad.SetLeftMargin(0.16)
            pad.SetRightMargin(0.0)
            r_pad.SetLeftMargin(0.16)
            r_pad.SetRightMargin(0.0)
        
        if i == 1:
            pad.SetRightMargin(0.0)
            pad.SetLeftMargin(0.0)
            r_pad.SetRightMargin(0.0)
            r_pad.SetLeftMargin(0.0)

        if i == 2:
            pad.SetRightMargin(0.1)
            pad.SetLeftMargin(0.0)
            r_pad.SetRightMargin(0.1)
            r_pad.SetLeftMargin(0.0)
        #else:
        #    pad.SetRightMargin(0.0)

        pad.Draw()
        r_pad.Draw()

    # =========================================================
    # Trova range globale Y
    # =========================================================

    global_min = 1.5*1e-3
    global_max = 2e-1
    x_axis_min = 5.0
    x_axis_max = 950

    datasets = []

    for fname in files:

        print(f"Processing {os.path.basename(fname)}")

        is_mc = os.path.basename(fname).startswith("MC")
        with open(fname) as jf:
            data = json.load(jf)

        results = data["results"]
        particle = data["particle"]
        betap = results["betap"]
        print(f"{particle}")
        betap_vals = np.array(betap, dtype=float)

        if particle == "carbon":
            betap_vals /= 12

        dataset = {
            "particle": particle,
            "layers": {},
            "is_mc": is_mc
        }

        for meth in methods:
            dataset["layers"][meth] = {}
            for lay in layers:
                dataset["layers"][meth][lay] = {}
                x = np.array(results[meth][lay]["x_std_dev"], dtype=float)
                xerr = np.array(results[meth][lay]["x_std_err"], dtype=float)
                y = np.array(results[meth][lay]["y_std_dev"], dtype=float)
                yerr = np.array(results[meth][lay]["y_std_err"], dtype=float)

                n = min(len(betap_vals), len(x), len(xerr))

                # Stampa prima dell'ordinamento
                print(f"\n\n{'='*50}")
                print(f"Layer {lay.upper()} - BEFORE sorting:")
                print(f"  betap: {betap_vals}")
                print(f"  x:     {x}")
                print(f"  xerr:  {xerr}")
                print(f"  y:     {y}")
                print(f"  yerr:  {yerr}")

                # Ordina
                points = sorted(zip(betap_vals, x, xerr, y, yerr), key=lambda t: t[0])
                betap_points, xvals, xerrs, yvals, yerrs = map(np.array, zip(*points))

                # Stampa dopo l'ordinamento
                print(f"\nLayer {lay.upper()} - AFTER sorting:")
                print(f"  betap: {betap_points}")
                print(f"  x:     {xvals}")
                print(f"  xerr:  {xerrs}")
                print(f"  y:     {yvals}")
                print(f"  yerr:  {yerrs}")
                print(f"{'='*50}")

                # Verifica che siano allineati
                for i in range(len(betap_points)):
                    print(f"  Point {i}: betap={betap_points[i]:.2f}, x={xvals[i]:.4f}, y={yvals[i]:.4f}")

                #ymin = np.min(yvals - yerrs)
                #ymax = np.max(yvals + yerrs)

                #if ymin > 0:
                #    global_min = min(global_min, ymin)
#
                #global_max = max(global_max, ymax)

                dataset["layers"][meth][lay] = {
                    "betap_points": betap_points,
                    "x": xvals,
                    "xerr": xerrs,
                    "y": yvals,
                    "yerr": yerrs
                }

        datasets.append(dataset)

    #global_min *= 0.8
    #global_max *= 1.2

    all_graphs = []
    all_bands = []
    all_ratio_graphs = []
    all_ratio_bands = []

    # =========================================================
    # Legenda unica
    # =========================================================

    legend = ROOT.TLegend(0.79, 0.72, 0.99, 0.93)

    legend.SetBorderSize(1)
    #legend.SetFillStyle(1001)
    legend.SetTextSize(0.035)
    #legend.SetTextFont(42)
    legend.SetNColumns(2)

    # =========================================================
    # Disegno
    # =========================================================

    for ilay, lay in enumerate(layers):

        pad = pads[ilay]
        pad.cd()

        first_graph = True

        for ifile, dataset in enumerate(datasets):

            bp = dataset["layers"]["m2"][lay]["betap_points"]
            x = dataset["layers"]["m2"][lay]["x"]
            xerr = dataset["layers"]["m2"][lay]["xerr"]
            y = dataset["layers"]["m2"][lay]["y"]
            yerr = dataset["layers"]["m2"][lay]["yerr"]

            n = len(bp)

            graph_x = ROOT.TGraphErrors(n)
            graph_x.SetTitle(f"Layer {lay.upper()}")
            band_x = ROOT.TGraphErrors(n)
            graph_y = ROOT.TGraphErrors(n)
            band_y = ROOT.TGraphErrors(n)

            for j in range(n):

                graph_x.SetPoint(j, bp[j], x[j])
                #graph_x.SetPointError(j, 0, xerr[j])
                band_x.SetPoint(j, bp[j], x[j])
                #low_err = min(xerr[j], x[j] - 1e-12)
                band_x.SetPointError(
                    j,
                    0,
                    xerr[j]
                )

                graph_y.SetPoint(j, bp[j], y[j])
                #graph_y.SetPointError(j, 0, yerr[j])
                band_y.SetPoint(j, bp[j], y[j])
                #low_err = min(yerr[j], y[j] - 1e-12)
                band_y.SetPointError(
                    j,
                    0,
                    yerr[j]
                )
                if dataset["is_mc"]:
                    graph_x.SetPointError(j, 0, 0)
                    graph_y.SetPointError(j, 0, 0)
                else:
                    graph_y.SetPointError(j, 0, yerr[j])
                    graph_x.SetPointError(j, 0, xerr[j])
            # =========================================
            # Style graph
            # =========================================

            graph_x.SetLineColor(colors[dataset["particle"]])
            graph_x.SetMarkerColor(colors[dataset["particle"]])

            graph_x.SetLineWidth(1)
            graph_x.SetLineStyle(2)

            graph_x.SetMarkerStyle(20)
            graph_x.SetMarkerSize(1.8)

            graph_y.SetLineColor(colors[dataset["particle"]])
            graph_y.SetMarkerColor(colors[dataset["particle"]])
            #graph_y.SetFillColorAlpha(colors[ifile], 0.60)

            graph_y.SetLineWidth(1)
            graph_y.SetLineStyle(2)
            graph_y.SetLineColorAlpha(colors[dataset["particle"]], 0.50)

            graph_y.SetMarkerStyle(26)
            graph_y.SetMarkerSize(1.8)

            # =========================================
            # Style band
            # =========================================

            band_x.SetFillColorAlpha(colors[dataset["particle"]], 0.25)
            band_x.SetLineColor(colors[dataset["particle"]])
            band_x.SetLineWidth(0)

            band_y.SetFillColorAlpha(colors[dataset["particle"]], 0.25)
            band_y.SetLineColor(colors[dataset["particle"]])
            band_y.SetLineWidth(0)

            # =========================================
            # Salva riferimenti
            # =========================================

            all_graphs.append(graph_x)
            all_bands.append(band_x)

            all_graphs.append(graph_y)
            all_bands.append(band_y)

            # =========================================
            # Primo draw
            # =========================================

            if first_graph:

                if dataset["is_mc"]:
                    graph_x.Draw("AL")
                else:   
                    graph_x.Draw("AP")

                graph_x.GetYaxis().SetRangeUser(global_min, global_max)

                graph_x.GetXaxis().SetLimits(
                    x_axis_min,
                    x_axis_max
                )

                # Nasconde label y nei pannelli centrali/destri
                if ilay != 0:
                    graph_x.GetYaxis().SetLabelSize(0)
                    graph_x.GetYaxis().SetTitleSize(0)

                graph_x.GetXaxis().SetLabelSize(0.053)
                graph_x.GetYaxis().SetLabelSize(0.053)

                graph_x.GetXaxis().SetTitleSize(0.05)
                graph_x.GetYaxis().SetTitleSize(0.05)

                graph_x.GetYaxis().SetTitleOffset(1.4)

                first_graph = False

            # =========================================
            # Disegno banda + linea
            # =========================================

            if dataset["is_mc"]:
                band_x.Draw("3 SAME")
                graph_x.Draw("L SAME")

                band_y.Draw("3 SAME")
                graph_y.Draw("L SAME")
            
            else:
                graph_x.Draw("P SAME")
                graph_y.Draw("P SAME")

            # legenda una volta sola
            if ilay == 0:
                if dataset['particle'] == "electron":
                    part = "e"
                elif dataset['particle'] == "proton":
                    part = "p"
                elif dataset['particle'] == "carbon":
                    part = "c"

                if dataset["is_mc"]:
                    legend.AddEntry(graph_x,
                                    f"{part} MC - x",
                                    "l")
                    legend.AddEntry(graph_y,
                                    f"{part} MC - y",
                                    "l")
                else:
                    legend.AddEntry(graph_x,
                                    f"{part} Data - x",
                                    "p")
                    legend.AddEntry(graph_y,
                                    f"{part} Data - y",
                                    "p")
                    
    for ilay, lay in enumerate(layers):

        pad = ratio_pads[ilay]
        pad.cd()

        first_graph = True

        for ifile, dataset in enumerate(datasets):

            bp = dataset["layers"]["m1"][lay]["betap_points"]

            x_m1 = dataset["layers"]["m1"][lay]["x"]
            xerr_m1 = dataset["layers"]["m1"][lay]["xerr"]
            y_m1 = dataset["layers"]["m1"][lay]["y"]
            yerr_m1 = dataset["layers"]["m1"][lay]["yerr"]

            x_m2 = dataset["layers"]["m2"][lay]["x"]
            xerr_m2 = dataset["layers"]["m2"][lay]["xerr"]
            y_m2 = dataset["layers"]["m2"][lay]["y"]
            yerr_m2 = dataset["layers"]["m2"][lay]["yerr"]

            n = len(bp)

            graph_x = ROOT.TGraphErrors(n)
            graph_x.SetTitle(f"")
            band_x = ROOT.TGraphErrors(n)
            graph_y = ROOT.TGraphErrors(n)
            band_y = ROOT.TGraphErrors(n)

            for j in range(n):

                
                x_ratio = x_m2[j]/x_m1[j]
                y_ratio = y_m2[j]/y_m1[j]
                print(f"particle: {dataset['particle']}, layer: {lay}, betap: {bp[j]}")
                #print(f"Layer: {lay} - betap: {bp[j]}")
                print(f"  x_std_m2: {x_m2[j]}, x_std_m1: {x_m1[j]}, ratio: {x_ratio}")
                print(f"  y_std_m2: {y_m2[j]}, y_std_m1: {y_m1[j]}, ratio: {y_ratio}\n")


                graph_x.SetPoint(j, bp[j], x_ratio)
                #graph_x.SetPointError(j, 0, x_ratio * np.sqrt((xerr_m2[j] / x_m2[j])**2 + (xerr_m1[j] / x_m1[j])**2))
                band_x.SetPoint(j, bp[j], x_ratio)
                #low_err = min(xerr[j], x[j] - 1e-12)
                band_x.SetPointError(j, 0, x_ratio * np.sqrt((xerr_m2[j] / x_m2[j])**2 + (xerr_m1[j] / x_m1[j])**2))

                graph_y.SetPoint(j, bp[j], y_ratio)
                #graph_y.SetPointError(j, 0, y_m2[j]/y_m1[j] * np.sqrt((yerr_m2[j] / y_m2[j])**2 + (yerr_m1[j] / y_m1[j])**2))
                band_y.SetPoint(j, bp[j], y_ratio)
                #low_err = min(xerr[j], x[j] - 1e-12)
                band_y.SetPointError(j, 0, y_ratio * np.sqrt((yerr_m2[j] / y_m2[j])**2 + (yerr_m1[j] / y_m1[j])**2))

                if dataset["is_mc"]:
                    graph_x.SetPointError(j, 0, 0)
                    graph_y.SetPointError(j, 0, 0)
                else:
                    graph_y.SetPointError(j, 0, y_ratio * np.sqrt((yerr_m2[j] / y_m2[j])**2 + (yerr_m1[j] / y_m1[j])**2))
                    graph_x.SetPointError(j, 0, x_ratio * np.sqrt((xerr_m2[j] / x_m2[j])**2 + (xerr_m1[j] / x_m1[j])**2))

            # =========================================
            # Style graph
            # =========================================

            graph_x.SetLineColor(colors[dataset["particle"]])
            graph_x.SetMarkerColor(colors[dataset["particle"]])

            graph_x.SetLineWidth(1)
            graph_x.SetLineStyle(2)

            graph_x.SetMarkerStyle(20)
            graph_x.SetMarkerSize(1.8)

            graph_y.SetLineColor(colors[dataset["particle"]])
            graph_y.SetMarkerColor(colors[dataset["particle"]])
            #graph_y.SetFillColorAlpha(colors[ifile], 0.60)

            graph_y.SetLineWidth(1)
            graph_y.SetLineStyle(2)
            graph_y.SetLineColorAlpha(colors[dataset["particle"]], 0.50)

            graph_y.SetMarkerStyle(26)
            graph_y.SetMarkerSize(1.8)

            # =========================================
            # Style band
            # =========================================

            band_x.SetFillColorAlpha(colors[dataset["particle"]], 0.25)
            band_x.SetLineColor(colors[dataset["particle"]])
            band_x.SetMarkerSize(0)
            band_x.SetLineWidth(0)

            band_y.SetFillColorAlpha(colors[dataset["particle"]], 0.25)
            band_y.SetLineColor(colors[dataset["particle"]])
            band_y.SetMarkerSize(0)
            band_y.SetLineWidth(0)

            # =========================================
            # Salva riferimenti
            # =========================================

            all_ratio_graphs.append(graph_x)
            all_ratio_bands.append(band_x)

            all_ratio_graphs.append(graph_y)
            all_ratio_bands.append(band_y)

            # =========================================
            # Primo draw
            # =========================================

            if first_graph:

                if dataset["is_mc"]:
                    graph_x.Draw("AL")
                else:   
                    graph_x.Draw("AP")

                graph_x.GetYaxis().SetRangeUser(0, 1.3)
                graph_x.GetYaxis().SetNdivisions(504)
                #graph_x.GetYaxis().SetLabelSize(0.1) 

                #graph_x.GetXaxis().SetLabelSize(0.12)   # Aumentato a 0.12
                #graph_x.GetYaxis().SetLabelSize(0.12)   # Aumentato a 0.12
                
                graph_x.GetXaxis().SetTitleSize(0.12)   # Aumentato a 0.12
                graph_x.GetYaxis().SetTitleSize(0.12)   # Aumentato a 0.12
                
                graph_x.GetYaxis().SetTitleOffset(0.8)  # Ridotto offset per dare più spazio
                
                # Forza anche il font
                #graph_x.GetXaxis().SetLabelFont(42)
                #graph_x.GetYaxis().SetLabelFont(42)

                graph_x.GetXaxis().SetLimits(
                    x_axis_min,
                    x_axis_max
                )

                graph_x.GetXaxis().SetLabelOffset(0.02)  # Leggero offset
                graph_x.GetYaxis().SetLabelOffset(0.02)
                graph_x.GetXaxis().SetTickLength(0.04)   # Lunghezza tacche

                # Forza il disegno delle tacche in alto
                graph_x.GetXaxis().SetAxisColor(ROOT.kBlack)
                graph_x.GetXaxis().SetLabelColor(ROOT.kBlack)

                # Le tacche sono già in basso di default, 
                # ma per averle anche in alto usa:
                pad = ratio_pads[ilay]
                pad.SetTickx(1)  # 1 = mostra tacche su entrambi i lati (alto e basso)
                #pad.SetTicky(1)

                # Nasconde label y nei pannelli centrali/destri
                if ilay != 0:
                    graph_x.GetYaxis().SetLabelSize(0)
                    graph_x.GetYaxis().SetTitleSize(0)

                graph_x.GetXaxis().SetLabelSize(0.12)
                graph_x.GetYaxis().SetLabelSize(0.12)

                graph_x.GetXaxis().SetTitleSize(0.05)
                graph_x.GetYaxis().SetTitleSize(0.05)

                graph_x.GetYaxis().SetTitleOffset(1.4)

                line = ROOT.TLine()
                line.SetLineStyle(2)  # Tratteggiata
                line.SetLineColor(ROOT.kGray + 1)
                line.SetLineWidth(2)

                # Disegna la linea orizzontale a y=1 su tutto il range x
                x_min = graph_x.GetXaxis().GetXmin()
                x_max = graph_x.GetXaxis().GetXmax()
                line.DrawLine(x_min, 1.0, x_max, 1.0)

                first_graph = False

            # =========================================
            # Disegno banda + linea
            # =========================================

            if dataset["is_mc"]:
                band_x.Draw("3 SAME")
                graph_x.Draw("L SAME")

                band_y.Draw("3 SAME")
                graph_y.Draw("L SAME")
            
            else:
                graph_x.Draw("P SAME")
                graph_y.Draw("P SAME")


    # =========================================================
    # Label globali
    # =========================================================

    c.cd()
    c.Update()

    xlabel = ROOT.TLatex()

    xlabel.SetNDC()
    xlabel.SetTextSize(0.042)
    xlabel.SetTextFont(42)

    xlabel.DrawLatex(
        0.45,
        0.03,
        "#betap/Z [MeV]"
    )

    ylabel1 = ROOT.TLatex()

    ylabel1.SetNDC()
    ylabel1.SetTextAngle(90)
    ylabel1.SetTextSize(0.042)
    ylabel1.SetTextFont(42)

    ylabel1.DrawLatex(
        0.020,
        0.52,
        "Residual #sigma [mm]"
    )

    ylabel2 = ROOT.TLatex()

    ylabel2.SetNDC()
    ylabel2.SetTextAngle(90)
    ylabel2.SetTextSize(0.042)
    ylabel2.SetTextFont(42)

    ylabel2.DrawLatex(
        0.020,
        0.09,
        "Comb. / Hough"
    )

    legend.Draw()

    c.Update()

    outname = os.path.join(
        output_dir,
        "m2_ratio_residuals_e_p_c.pdf"
    )

    c.SaveAs(outname)

    print(f"\nSaved plot to:")
    print(outname)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Residual plots"
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

    res_plots(
        args.input_dir,
        args.output_dir
    )