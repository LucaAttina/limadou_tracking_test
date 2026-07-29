import ROOT
import numpy as np
import os
import argparse
import sys
import re
import json

legends = []

def analyze_histogram_properties(values, energy):
    """
    Analyze data to define adequate range, number of bins and parameters.
    """
    if not values or len(values) == 0:
        return None
    
    values = np.array(values)
    mean = np.mean(values)
    std_dev = np.std(values)
    mask = np.abs(values - mean) < 3 * std_dev
    filtered_values = values[mask]
    std_dev = np.std(filtered_values)
    x_min = values.min()
    x_max = values.max()
    
    if energy < 20:
        n_bins = 12
    else: 
        n_bins = 20
    
    return {
        'mean': mean,
        'std_dev': std_dev,
        'x_min': x_min,
        'x_max': x_max,
        'n_bins': n_bins,
        'n_entries': len(values)
    }


def initialize_std_results():
    """Initialize the structure to store fit results (sigma and errors)"""
    return {
        "betap": [],
        "m1": {
            "l1": {"x_rms": [], "y_rms": [], "x_rms_err": [], "y_rms_err": []},
            "l2": {"x_rms": [], "y_rms": [], "x_rms_err": [], "y_rms_err": []},
            "l3": {"x_rms": [], "y_rms": [], "x_rms_err": [], "y_rms_err": []},
        },
        "m2": {
            "l1": {"x_rms": [], "y_rms": [], "x_rms_err": [], "y_rms_err": []},
            "l2": {"x_rms": [], "y_rms": [], "x_rms_err": [], "y_rms_err": []},
            "l3": {"x_rms": [], "y_rms": [], "x_rms_err": [], "y_rms_err": []},
        }
    }


def add_rms_results(std_results, betap, method, layer, coord, hist, values):
    """Add RMS results to the results structure."""
    if hist is None or hist.GetEntries() == 0:
        return
    
    try:
        
        values = np.array(values)
        mean = np.mean(values)
        std = np.std(values)
        
        # Filtra entro 3 sigma
        mask = np.abs(values - mean) < 3 * std
        filtered_values = values[mask]
        
        # Calcola RMS sui dati filtrati
        sigma = np.std(filtered_values)
        sigma_err = sigma / np.sqrt(len(filtered_values))

        sigma_tot = hist.GetRMS()
        #sigma_err = hist.GetRMSError()
        
        layer_str = f"l{layer}"
        method_str = f"m{method}"
        coord_key_dev = f"{coord}_rms"
        coord_key_err = f"{coord}_rms_err"
        
        std_results[method_str][layer_str][coord_key_dev].append(sigma)
        std_results[method_str][layer_str][coord_key_err].append(sigma_err)
        print(f"   📊 {method_str} L{layer} {coord}: RMS completo={sigma_tot:.4f}, RMS 3σ={sigma:.4f} ({len(filtered_values)}/{len(values)} entries)")
    except Exception as e:
        print(f"⚠️ Error adding RMS results: {e}")


def load_all_trees_data(root_file_path):
    """Read ALL trees from the ROOT file."""
    f = ROOT.TFile(root_file_path, "READ")
    
    trees = []
    for key in f.GetListOfKeys():
        obj = key.ReadObj()
        if obj.InheritsFrom("TTree"):
            tree_name = key.GetName()
            trees.append((tree_name, obj))
            print(f"📂 Found tree: {tree_name}")
    
    if not trees:
        raise RuntimeError(f"No TTree found in file {root_file_path}")
    
    all_data = {}
    
    for tree_name, tree in trees:
        print(f"\n📖 Processing tree: {tree_name}")
        
        tree_data = {
            'energy': None,
            'data': {}
        }
        
        total_entries = 0
        
        for entry in tree:
            energy = entry.energy
            betap = entry.betap
            method = entry.method
            layer = entry.layer
            res_x = entry.res_x
            res_y = entry.res_y
            
            if tree_data['energy'] is None:
                tree_data['energy'] = energy
            
            tree_data['data'].setdefault(betap, {}).setdefault(method, {}).setdefault(layer, {'x': [], 'y': []})
            tree_data['data'][betap][method][layer]['x'].append(res_x)
            tree_data['data'][betap][method][layer]['y'].append(res_y)
            total_entries += 1
        
        all_data[tree_name] = tree_data
        
        print(f"   Energy: {tree_data['energy']:.0f} MeV")
        print(f"   Total entries: {total_entries}")
        
        for betap in tree_data['data'].keys():
            n_events = sum(len(tree_data['data'][betap][method][layer]['x']) 
                           for method in tree_data['data'][betap] 
                           for layer in tree_data['data'][betap][method])
            print(f"   beta p={betap:.4f}: {n_events} entries")
    
    f.Close()
    return all_data


def create_histogram_adaptive(values, name, title, energy):
    """Create a TH1F from a list of values with adaptive x range and number of bins"""
    if not values or len(values) == 0:
        return None
    
    n_bins = 40
    if energy >= 50 and "M2" in title:
        n_bins -= 10
    props = analyze_histogram_properties(values, energy)
    h = ROOT.TH1F(name, title, n_bins, props["mean"]-3*props["std_dev"], props["mean"]+3*props["std_dev"])
    
    for v in values:
        h.Fill(v)
    
    return h, props


def save_histograms_to_root(all_histograms, output_file):
    """Save all histograms to a ROOT file for later inspection."""
    root_file = ROOT.TFile(output_file, "RECREATE")
    hist_count = 0
    
    for tree_name, tree_hists in all_histograms.items():
        tree_dir = root_file.mkdir(f"tree_{tree_name}")
        tree_dir.cd()
        
        for betap, betap_hists in tree_hists.items():
            betap_dir = tree_dir.mkdir(f"betap_{betap:.6f}")
            betap_dir.cd()
            
            for method, method_hists in betap_hists.items():
                method_dir = betap_dir.mkdir(f"method_{method}")
                method_dir.cd()
                
                for layer, layer_hists in method_hists.items():
                    layer_dir = method_dir.mkdir(f"layer_{layer}")
                    layer_dir.cd()
                    
                    for coord, hist in layer_hists.items():
                        if hist is not None:
                            hist.Write()
                            hist_count += 1
    
    root_file.Close()
    print(f"\n✅ Saved {hist_count} histograms to {output_file}")


def draw_histogram_with_legend(hist, props, title, pad, coord_name):
    """Draw a histogram with a legend showing mean and std dev."""
    global legends
    pad.cd()
    #pad.SetGrid()
    pad.SetMargin(0.12, 0.04, 0.12, 0.08)
    
    if hist and hist.GetEntries() > 0:
        # Style the histogram
        hist.SetTitle(title)
        hist.GetXaxis().SetTitle("Residual [mm]")
        hist.GetYaxis().SetTitle("Entries")
        hist.SetLineColor(ROOT.kRed)
        hist.SetLineWidth(1)
        #hist.SetFillStyle(3001)
        #hist.SetFillColor(ROOT.kGray)
        # Prima di Draw
        xmin = props['mean'] - 3*props['std_dev']
        xmax = props['mean'] + 3*props['std_dev']
        hist.GetXaxis().SetRangeUser(xmin, xmax)
        #print(f"AAAA: {xmin} - {xmax}")
        hist.Draw("E")
        ROOT.gStyle.SetOptStat(0)
        
        pad.cd()
        # Create legend
        legend = ROOT.TLegend(0.67, 0.72, 0.96, 0.92)
        #legend.SetBorderSize(0)
        legend.SetFillStyle(1001)
        legend.SetFillColor(ROOT.kWhite)
        legend.SetTextSize(0.034)
        legend.SetFillStyle(0)
        legend.SetEntrySeparation(0.0)  # Rimuove la separazione tra simbolo e testo
        legend.SetMargin(0.05)           # Rimuove il margine sinistro
        #legend.SetTextSize(0.04)
        
        # Add entries
        mean = hist.GetMean()
        mean_err = hist.GetMeanError()

        #print(f"Mean = {mean:.5f} ± {mean_err:.5f}")
        legend.AddEntry(ROOT.nullptr, f"Entries: {int(hist.GetEntries())}", "")
        legend.AddEntry(ROOT.nullptr, f"Mean: ({mean:.4f}#pm{mean_err:.4f}) mm", "")
        #legend.AddEntry("", f"Std Dev: {props['std_dev']:.4f} mm", "")
        
        # Add RMS from histogram
        rms = hist.GetRMS()
        rms_err = hist.GetRMSError()
        legend.AddEntry(ROOT.nullptr, f"RMS: ({rms:.4f}#pm{rms_err:.4f}) mm", "")
        
        legend.Draw()
        legends.append(legend)
        pad.Update()
        
        # Add coordinate label
        #label = ROOT.TLatex()
        #label.SetTextAlign(12)
        #label.SetTextSize(0.04)
        #label.DrawLatex(0.15, 0.92, f"{coord_name}")
        
    else:
        # Empty histogram
        dummy = ROOT.TH1F("dummy", title, 10, -5, 5)
        dummy.SetFillStyle(0)
        dummy.GetXaxis().SetTitle("Residual [mm]")
        dummy.GetYaxis().SetTitle("Counts")
        dummy.SetMaximum(1)
        dummy.SetMinimum(0)
        dummy.Draw()
        
        label = ROOT.TLatex()
        label.SetTextAlign(22)
        label.SetTextSize(0.05)
        label.DrawLatex(0, 0.5, "No data available")


def process_single_tree(tree_name, tree_data, output_base_dir, mc):
    """Process a single tree (one energy): create histograms and produce PDF with all layers."""
    energy = tree_data['energy']
    data = tree_data['data']
    
    print(f"\n{'='*60}")
    print(f"📁 Processing Tree: {tree_name}")
    print(f"   Energy = {energy:.0f} MeV")
    print(f"{'='*60}")
    
    if not mc:
        output_pdf = f"{output_base_dir}/debug_plots/{tree_name}.pdf"
    else:
        output_pdf = f"{output_base_dir}/debug_plots/MC_{tree_name}.pdf"
    tree_histograms = {}
    std_results = initialize_std_results()
    
    betap_values = sorted(data.keys())
    
    # Get all available layers
    all_layers = set()
    for betap in betap_values:
        for method in data[betap].keys():
            all_layers.update(data[betap][method].keys())
    layers = sorted(all_layers)
    
    print(f"📊 Found layers: {layers}")
    
    # Create canvas for PDF
    canvas = ROOT.TCanvas(f"c_{tree_name}", f"Energy = {energy:.0f} MeV", 1200, 800)  # Taller canvas for 3 layers
    
    first_page = True
    
    for betap in betap_values:
        tree_histograms[betap] = {1: {}, 2: {}}
        
        if betap not in std_results["betap"]:
            std_results["betap"].append(betap)
        
        # For each layer, create a separate page with 2x2 layout
        for layer in layers:
            print(f"\n  📊 Creating page for beta={betap:.6f}, layer={layer}")
            
            # Verify data exists for this layer and both methods
            if 1 not in data[betap] or 2 not in data[betap]:
                print(f"⚠️ beta={betap:.4f}: method 1 or 2 missing")
                continue
            if layer not in data[betap][1] or layer not in data[betap][2]:
                print(f"⚠️ beta={betap:.4f}, layer={layer}: missing data")
                continue
            
            # Create histograms
            suffix = f"betap{betap:.4f}_L{layer}"
            
            h_x_m1, props_x_m1 = create_histogram_adaptive(
                data[betap][1][layer]['x'],
                f"h_x_m1_{suffix}",
                f"X residuals - M1, layer {layer}",
                energy
            )
            h_y_m1, props_y_m1 = create_histogram_adaptive(
                data[betap][1][layer]['y'],
                f"h_y_m1_{suffix}",
                f"Y residuals - M1, layer {layer}",
                energy
            )
            h_x_m2, props_x_m2 = create_histogram_adaptive(
                data[betap][2][layer]['x'],
                f"h_x_m2_{suffix}",
                f"X residuals - M2, layer {layer}",
                energy
            )
            h_y_m2, props_y_m2 = create_histogram_adaptive(
                data[betap][2][layer]['y'],
                f"h_y_m2_{suffix}",
                f"Y residuals - M2, layer {layer}",
                energy
            )
            
            # Store histograms
            tree_histograms[betap][1][layer] = {'x': h_x_m1, 'y': h_y_m1}
            tree_histograms[betap][2][layer] = {'x': h_x_m2, 'y': h_y_m2}
            
            # Add RMS results
            add_rms_results(std_results, betap, 1, layer, 'x', h_x_m1, data[betap][1][layer]['x'])
            add_rms_results(std_results, betap, 1, layer, 'y', h_y_m1, data[betap][1][layer]['y'])
            add_rms_results(std_results, betap, 2, layer, 'x', h_x_m2, data[betap][2][layer]['x'])
            add_rms_results(std_results, betap, 2, layer, 'y', h_y_m2, data[betap][2][layer]['y'])

            for meth in [1,2]:
                for coord in ["x","y"]:
                    rms_vals = std_results[f"m{meth}"][f"l{layer}"][f"{coord}_rms"][-1]
                    std_vals = np.std(data[betap][meth][layer][coord])

                    print(f"m{meth} L{layer} {coord}: RMS = {rms_vals:.4f} | STD = {std_vals:.4f}")

            for coord in ["x", "y"]:
                key = f"{coord}_rms"

                m1_vals = std_results["m1"][f"l{layer}"][key]
                m2_vals = std_results["m2"][f"l{layer}"][key]

                if len(m1_vals) == 0 or len(m2_vals) == 0:
                    continue

                std_m1 = m1_vals[-1]
                std_m2 = m2_vals[-1]

                data_m1 = np.std(data[betap][1][layer][coord])
                data_m2 = np.std(data[betap][2][layer][coord])

                ratio = std_m2 / std_m1 if std_m1 != 0 else float("inf")

                print(
                    f"L{layer} {coord}: "
                    f"RMS M1 = {std_m1:.4f} | M2 = {std_m2:.4f} | "
                    f"ratio M2/M1 = {ratio:.4f}"
                )

            '''
            values = data[betap][1][layer]['x']
            std_results["m1"][f"l{layer}"]["x_rms"].append(np.std(values))
            std_results["m1"][f"l{layer}"]["x_rms_err"].append(np.std(values)/np.sqrt(len(values)))

            values = data[betap][1][layer]['y']
            std_results["m1"][f"l{layer}"]["y_rms"].append(np.std(values))
            std_results["m1"][f"l{layer}"]["y_rms_err"].append(np.std(values)/np.sqrt(len(values)))

            values = data[betap][2][layer]['x']
            std_results["m2"][f"l{layer}"]["x_rms"].append(np.std(values))
            std_results["m2"][f"l{layer}"]["x_rms_err"].append(np.std(values)/np.sqrt(len(values)))

            values = data[betap][2][layer]['y']
            std_results["m2"][f"l{layer}"]["y_rms"].append(np.std(values))
            std_results["m2"][f"l{layer}"]["y_rms_err"].append(np.std(values)/np.sqrt(len(values)))
            '''
            
            # Create 2x2 layout for this layer
            canvas.cd()
            canvas.Clear()
            canvas.Divide(2, 2)
            
            # Draw histograms with legends
            # Pad 1
            pad1 = canvas.cd(1)
            draw_histogram_with_legend(h_x_m1, props_x_m1,
                                    f"X residuals - M1, layer {layer} - #betap = {betap:.2f} MeV",
                                    pad1,  # Passa il pad, non canvas.cd(1)
                                    "X - M1")

            # Pad 2
            pad2 = canvas.cd(2)
            draw_histogram_with_legend(h_y_m1, props_y_m1,
                                    f"Y residuals - M1, layer {layer} - #betap = {betap:.2f} MeV",
                                    pad2,
                                    "Y - M1")

            # Pad 3
            pad3 = canvas.cd(3)
            draw_histogram_with_legend(h_x_m2, props_x_m2,
                                    f"X residuals - M2, layer {layer} - #betap = {betap:.2f} MeV",
                                    pad3,
                                    "X - M2")

            # Pad 4
            pad4 = canvas.cd(4)
            draw_histogram_with_legend(h_y_m2, props_y_m2,
                                    f"Y residuals - M2, layer {layer} - #betap = {betap:.2f} MeV",
                                    pad4,
                                    "Y - M2")

            # IMPORTANTE: Aggiorna il canvas DOPO aver disegnato tutto
            canvas.Update()
            canvas.Modified()
            
            # Add title at the top
            #canvas.cd(0)
            #title_obj = ROOT.TLatex()
            #title_obj.SetTextAlign(22)
            #title_obj.SetTextSize(0.025)
            #title_obj.DrawLatex(0.5, 0.97, 
                               #f"Tree: {tree_name} | Energy = {energy:.0f} MeV | #beta p = {betap:.6f} | Layer {layer}")
            
            #canvas.Update()
            
            # Save page in PDF
            if first_page:
                canvas.Print(output_pdf + "(")
                first_page = False
            else:
                canvas.Print(output_pdf)

            
            print(f"    ✅ Page saved for Layer {layer}")
    
        if not first_page:
            canvas.Print(output_pdf + ")")
    
    print(f"\n📄 PDF created: {output_pdf}\n")
    return tree_histograms, std_results


def process_all_trees(input_file, output_dir, particle, mc):
    """Process ALL trees in the ROOT file."""
    print(f"\n📖 Reading file: {input_file}")
    all_data = load_all_trees_data(input_file)
    
    if not all_data:
        print(f"❌ Error: No trees found in file {input_file}")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_histograms = {}
    
    all_std_results = {
        "betap": [],
        "m1": {
            "l1": {"x_rms": [], "y_rms": [], "x_rms_err": [], "y_rms_err": []},
            "l2": {"x_rms": [], "y_rms": [], "x_rms_err": [], "y_rms_err": []},
            "l3": {"x_rms": [], "y_rms": [], "x_rms_err": [], "y_rms_err": []},
        },
        "m2": {
            "l1": {"x_rms": [], "y_rms": [], "x_rms_err": [], "y_rms_err": []},
            "l2": {"x_rms": [], "y_rms": [], "x_rms_err": [], "y_rms_err": []},
            "l3": {"x_rms": [], "y_rms": [], "x_rms_err": [], "y_rms_err": []},
        }
    }
    
    print(f"\n{'='*60}")
    print(f"🎯 Processing {len(all_data)} trees found")
    print(f"{'='*60}")
    
    for tree_name, tree_data in all_data.items():
        tree_histograms, std_results = process_single_tree(tree_name, tree_data, output_dir, mc)
        all_histograms[tree_name] = tree_histograms
        
        # Merge results
        for method in ["m1", "m2"]:
            for layer in ["l1", "l2", "l3"]:
                for coord in ["x", "y"]:
                    all_std_results[method][layer][f"{coord}_rms"].extend(
                        std_results[method][layer][f"{coord}_rms"]
                    )
                    all_std_results[method][layer][f"{coord}_rms_err"].extend(
                        std_results[method][layer][f"{coord}_rms_err"]
                    )
        
        for betap in std_results["betap"]:
            if betap not in all_std_results["betap"]:
                all_std_results["betap"].append(betap)
    
    base = os.path.basename(input_file)
    prefix = base.split("residuals")[0].rstrip("_")
    
    # Save histograms to ROOT file (optional - commented out)
    # histograms_output = os.path.join(output_dir, f"{prefix}_residuals_histograms.root")
    # save_histograms_to_root(all_histograms, histograms_output)
    
    if particle == "e":
        part = "electron"
    elif particle == "p":
        part = "proton"
    elif particle == "c":
        part = "carbon"
    
    # Save results to JSON
    results_output = os.path.join(output_dir, f"{prefix}_residuals_results.json")
    with open(results_output, 'w') as f:
        fin_data = {
            "particle": part,
            "results": all_std_results
        }
        json.dump(fin_data, f, indent=4)
    print(f"✅ Fit results saved to: {results_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit residuals distributions from ROOT file with multiple trees")
    parser.add_argument("--input-file", required=True, 
                        help="Input ROOT file containing multiple residuals trees")
    parser.add_argument("--output-dir", required=True, 
                        help="Output directory for fit results and PDFs")
    parser.add_argument("--particle", required=True, help="Select particle configuration.")
    parser.add_argument("--mc", action="store_true", help="Define name.")


    args = parser.parse_args()
    ROOT.EnableImplicitMT(20) 
    
    if args.particle not in ["e", "p", "c"]:
        print(f"Error: Invalid particle type '{args.particle}'. Must be one of: e, p, c.")
        sys.exit(1)
    
    if not os.path.exists(args.input_file):
        print(f"❌ Error: File {args.input_file} does not exist!")
        sys.exit(1)
    
    print("="*60)
    print("🔧 RESIDUALS FITTER - MULTIPLE TREES")
    print("="*60)
    print(f"📂 Input file:  {args.input_file}")
    print(f"📁 Output dir:  {args.output_dir}")
    print("="*60)
    
    process_all_trees(args.input_file, args.output_dir, args.particle, args.mc)
    
    print("\n" + "="*60)
    print("✅ COMPLETED!")
    print(f"📁 Results saved in: {args.output_dir}/")
    print("="*60)