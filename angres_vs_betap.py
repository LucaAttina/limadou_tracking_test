import argparse
import os
import numpy as np
import ROOT
import glob
import json
import sys


def res_plots(input_dir, output_dir, value):

    files = sorted(glob.glob(os.path.join(input_dir, f"*{value}.json")))

    if not files:
        raise RuntimeError(f"No .json files found in {input_dir}")

    print("\nFiles to be processed:")
    print("-" * 60)
    for f in files:
        print(os.path.basename(f))
    print("-" * 60 + "\n")

    ROOT.gStyle.SetOptStat(0)
    #ROOT.gStyle.SetTitleFont(42, "")          
    ROOT.gStyle.SetTitleSize(0.06, "Angular resolution VS #betap")  

    #layers = ["l1", "l2", "l3"]
    methods = ["m1", "m2"]

    colors = {
        "e": ROOT.kBlack,
        "p": ROOT.kRed,
        "c": ROOT.kGreen+3
    }

    # =========================================================
    # Canvas
    # =========================================================

    c = ROOT.TCanvas("c", "Resolution", 2000, 1100)

    # Pad attaccati
    pad1 = ROOT.TPad("pad1", "", 0.00, 0.30, 1.00, 1.00)
    pad2 = ROOT.TPad("pad2", "", 0.00, 0.00, 1.00, 0.30)
    #pad3 = ROOT.TPad("pad3", "", 0.666, 0.30, 1.00, 1.00)
#
    #pad4 = ROOT.TPad("pad4", "", 0.00, 0.00, 0.35, 0.30)
    #pad5 = ROOT.TPad("pad5", "", 0.35, 0.00, 0.666, 0.30)
    #pad6 = ROOT.TPad("pad6", "", 0.666, 0.00, 1.00, 0.30)

    pads = [pad1]
    ratio_pads = [pad2]

    for i, (pad, r_pad) in enumerate(zip(pads, ratio_pads)):
    
        pad.SetLogx()
        r_pad.SetLogx()

        
        pad.SetTopMargin(0.1)
        pad.SetBottomMargin(0)
        pad.SetRightMargin(0.05)
        r_pad.SetTopMargin(0)
        r_pad.SetBottomMargin(0.27)
        r_pad.SetRightMargin(0.05)

        pad.Draw()
        r_pad.Draw()


    # =========================================================
    # Trova range globale Y
    # =========================================================

    global_min = -2
    global_max = 50
    x_axis_min = 5.0
    x_axis_max = 950
    c.SetLogx()
    

    datasets = []

    for fname in files:

        pads[0].cd()
        print(f"Processing {os.path.basename(fname)}")

        with open(fname) as jf:
            data = json.load(jf)

        particle = data["particle"]
        is_mc = data["MC"]

        print(f"Particle: {particle}, MC: {is_mc}")

        trees = data["trees"]

        dataset = {
            "particle": particle,
            "is_mc": is_mc,
            "betap": [],
            "m1": {
                "val": [],
                "err": [],
                "rms": []
            },
            "m2": {
                "val": [],
                "err": [],
                "rms": []
            }
        }


        # loop sui tree (energie diverse)
        for tree_name, tree_data in trees.items():

            betap = tree_data["betap"][0]

            # eventuale correzione carbonio
            if particle == "carbon":
                betap /= 12.

            dataset["betap"].append(betap)

            for meth in ["m1", "m2"]:

                for quantity in ["val", "err", "rms"]:

                    num = tree_data[meth][quantity][0]

                    dataset[meth][quantity].append(num)


        # ordino per betap crescente
        order = np.argsort(dataset["betap"])

        dataset["betap"] = np.array(dataset["betap"])[order]


        for meth in ["m1", "m2"]:
            for quantity in ["val", "err", "rms"]:

                dataset[meth][quantity] = np.array(
                    dataset[meth][quantity]
                )[order]


        datasets.append(dataset)

    #global_min *= 0.8
    #global_max *= 1.2

    all_graphs = []
    all_bands = []
    all_ratio_graphs = []
    all_ratio_bands = []

    # =========================================================
    # Disegno
    # =========================================================

    first_graph = True

    for dataset in datasets:

        bp = dataset["betap"]
        if value == "peak":
            val = dataset["m2"]["val"] # mpv
            err = dataset["m2"]["err"] # fwhm
            err = err/2 # hwhm
        elif value == "hwhm":
            val = dataset["m2"]["val"] # hwhm
            err = dataset["m2"]["err"] # semidifference hwhm 90%-10%          

        n = len(bp)

        particle = dataset["particle"]
        color = colors[particle]

        graph_x = ROOT.TGraphErrors(n)
        band_x = ROOT.TGraphErrors(n)

        for j in range(n):

            graph_x.SetPoint(j, bp[j], val[j])
            #graph_x.SetPointError(j, 0, xerr[j])
            band_x.SetPoint(j, bp[j], val[j])
            #low_err = min(xerr[j], x[j] - 1e-12)
            band_x.SetPointError(
                j,
                0,
                err[j]
            )

            if dataset["is_mc"]:
                graph_x.SetPointError(j, 0, 0)
            else:
                graph_x.SetPointError(j, 0, err[j])

        # =========================================
        # Style graph
        # =========================================

        graph_x.SetLineColor(colors[dataset["particle"]])
        graph_x.SetMarkerColor(colors[dataset["particle"]])

        graph_x.SetLineWidth(1)
        if dataset["is_mc"]:
            graph_x.SetLineStyle(2)
        else:
            graph_x.SetLineStyle(1)

        graph_x.SetMarkerStyle(20)
        graph_x.SetMarkerSize(1.8)

        # =========================================
        # Style band
        # =========================================

        band_x.SetFillColorAlpha(colors[dataset["particle"]], 0.25)
        band_x.SetLineColor(colors[dataset["particle"]])
        band_x.SetLineWidth(0)

        # =========================================
        # Salva riferimenti
        # =========================================

        all_graphs.append(graph_x)
        all_bands.append(band_x)

        # =========================================
        # Primo draw
        # =========================================

        if first_graph:

            if dataset["is_mc"]:
                graph_x.SetTitle("")

                graph_x.Draw("AL")
            else:   
                graph_x.Draw("AP")

            graph_x.GetYaxis().SetRangeUser(global_min, global_max)

            graph_x.GetXaxis().SetLimits(
                x_axis_min,
                x_axis_max
            )

            # Nasconde label y nei pannelli centrali/destri

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
        else:
            graph_x.Draw("P SAME")

        # legenda una volta sola
        if dataset['particle'] == "e":
            part = "e"
        elif dataset['particle'] == "p":
            part = "p"
        elif dataset['particle'] == "c":
            part = "C"
              
    ratio_pads[0].cd()

    first_graph = True

    for ifile, dataset in enumerate(datasets):

        bp = dataset["betap"]
        if value == "peak":
            val1 = dataset["m1"]["val"]
            err1 = dataset["m1"]["err"] / 2 # fwhm / 2
            val2 = dataset["m2"]["val"]
            err2 = dataset["m2"]["err"] / 2
        elif value == "hwhm":
            val1 = dataset["m1"]["val"]
            err1 = dataset["m1"]["err"]
            val2 = dataset["m2"]["val"]
            err2 = dataset["m2"]["err"]

        n = len(bp)

        particle = dataset["particle"]
        color = colors[particle]

        graph_x = ROOT.TGraphErrors(n)
        graph_x.SetTitle(f"")
        band_x = ROOT.TGraphErrors(n)

        print(f"{value}")

        for j in range(n):

            r_ratio = val2[j]/val1[j]
            r_err = r_ratio * np.abs((err2[j] / val2[j]) - (err1[j] / val1[j]))
            print(f"particle: {dataset['particle']}, betap: {bp[j]}")
            #print(f"Layer: {lay} - betap: {bp[j]}")
            print(f"  x_std_m2: {val2[j]}, x_std_m1: {val1[j]}, ratio: {r_ratio}")


            graph_x.SetPoint(j, bp[j], r_ratio)
            band_x.SetPoint(j, bp[j], r_ratio)
            #graph_x.SetPointError(j, 0, x_ratio * np.sqrt((xerr_m2[j] / x_m2[j])**2 + (xerr_m1[j] / x_m1[j])**2))
            #low_err = min(xerr[j], x[j] - 1e-12)
            band_x.SetPointError(j, 0, r_err)

            if dataset["is_mc"]:
                graph_x.SetPointError(j, 0, 0)
            else:
                graph_x.SetPointError(j, 0, r_err)

        # =========================================
        # Style graph
        # =========================================

        graph_x.SetLineColor(colors[dataset["particle"]])
        graph_x.SetMarkerColor(colors[dataset["particle"]])

        graph_x.SetLineWidth(1)
        #graph_x.SetLineStyle(2)
        if dataset["is_mc"]:
            graph_x.SetLineStyle(2)
        else:
            graph_x.SetLineStyle(1)

        graph_x.SetMarkerStyle(20)
        graph_x.SetMarkerSize(1.8)

        # =========================================
        # Style band
        # =========================================

        band_x.SetFillColorAlpha(colors[dataset["particle"]], 0.25)
        band_x.SetLineColor(colors[dataset["particle"]])
        band_x.SetMarkerSize(0)
        band_x.SetLineWidth(0)



        # =========================================
        # Salva riferimenti
        # =========================================

        all_ratio_graphs.append(graph_x)
        all_ratio_bands.append(band_x)

        # =========================================
        # Primo draw
        # =========================================

        if first_graph:

            if dataset["is_mc"]:
                graph_x.Draw("AL")
            else:   
                graph_x.Draw("AP")

            graph_x.GetYaxis().SetRangeUser(0, 1.7)
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
            graph_x.GetXaxis().SetTickLength(0.06)   # Lunghezza tacche

            # Forza il disegno delle tacche in alto
            graph_x.GetXaxis().SetAxisColor(ROOT.kBlack)
            graph_x.GetXaxis().SetLabelColor(ROOT.kBlack)

            # Le tacche sono già in basso di default, 
            # ma per averle anche in alto usa:
            pad.SetTickx(1)  # 1 = mostra tacche su entrambi i lati (alto e basso)
            #pad.SetTicky(1)

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
        
        else:
            graph_x.Draw("P SAME")

    # =========================================================
    # Label globali
    # =========================================================

    c.cd()
    c.Update()

    xlabel = ROOT.TLatex()

    xlabel.SetNDC()
    xlabel.SetTextSize(0.045)
    xlabel.SetTextFont(42)

    xlabel.DrawLatex(
        0.80,
        0.03,
        "#betap/Z [MeV]"
    )

    ylabel1 = ROOT.TLatex()

    ylabel1.SetNDC()
    ylabel1.SetTextAngle(90)
    ylabel1.SetTextSize(0.045)
    ylabel1.SetTextFont(42)

    ylabel1.DrawLatex(
        0.040,
        0.50,
        "Angular resolution [deg]"
    )

    
    ylabel2 = ROOT.TLatex()

    ylabel2.SetNDC()
    ylabel2.SetTextAngle(90)
    ylabel2.SetTextSize(0.042)
    ylabel2.SetTextFont(42)

    ylabel2.DrawLatex(
        0.040,
        0.09,
        "Comb. / Hough"
    )

    # =========================================================
    # Legenda - Forza l'ordine MC a sinistra, Data a destra
    # =========================================================

    legend = ROOT.TLegend(0.75, 0.74, 0.95, 0.93)
    legend.SetBorderSize(1)
    legend.SetTextSize(0.04)
    legend.SetNColumns(2)  # 2 colonne
    legend.SetMargin(0.3)

    particle_names = {"e": "e", "p": "p", "c": "C"}
    particle_order = ["e", "p", "c"]
    
    # Buffer per MC e Data separati
    mc_entries = []
    data_entries = []
    
    # Raccogli gli entry
    for dataset in datasets:
        particle = dataset["particle"]
        is_mc = dataset["is_mc"]
        
        part_label = particle_names[particle]
        label = f"{part_label} {'MC' if is_mc else 'Data'}"
        
        target_style = 2 if is_mc else 1
        target_graph = None
        
        for graph in all_graphs:
            if (graph.GetLineColor() == colors[particle] and 
                graph.GetLineStyle() == target_style):
                target_graph = graph
                break
        
        if target_graph is not None:
            if is_mc:
                mc_entries.append((particle_order.index(particle), target_graph, label, "l"))
            else:
                data_entries.append((particle_order.index(particle), target_graph, label, "p"))
    
    # Ordina entrambe le liste per particella
    mc_entries.sort(key=lambda x: x[0])
    data_entries.sort(key=lambda x: x[0])
    
    # Aggiungi alla legenda in questo ordine:
    # e MC, e Data, p MC, p Data, C MC, C Data
    # Questo con 2 colonne darà:
    # Colonna1: e MC, p MC, C MC
    # Colonna2: e Data, p Data, C Data
    for i in range(len(mc_entries)):
        # Aggiungi MC
        _, graph, label, option = mc_entries[i]
        legend.AddEntry(graph, label, option)
        # Aggiungi Data
        _, graph, label, option = data_entries[i]
        legend.AddEntry(graph, label, option)
    

    legend.Draw()

    c.Update()

    outname = os.path.join(
        output_dir,
        f"m2_angres_e_p_c_{value}.pdf"
    )

    c.SaveAs(outname)

    print(f"\nSaved plot to:")
    print(outname)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Angular resolution plots"
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

    parser.add_argument(
        "--value",
        required=True,
        help="Peak for mpv, hwhm otherwise"
    )
    

    args = parser.parse_args()

    if args.value not in ["peak", "hwhm"]:
        print(f"Error: Invalid method '{args.value}'. Must be one of: peak, hwhm.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    res_plots(
        args.input_dir,
        args.output_dir,
        args.value
    )