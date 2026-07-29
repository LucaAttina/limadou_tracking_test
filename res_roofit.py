import ROOT
import numpy as np
from roofitter import RooFitter
import os
import argparse
import sys
import re
import json

print(ROOT.IsImplicitMTEnabled())
print(ROOT.GetThreadPoolSize())

ROOT.EnableImplicitMT(20)

print(ROOT.GetThreadPoolSize())



def analyze_histogram_properties(values, energy, title, mc, particle):
    """
    Analyze data to define adequate range, number of bins and parameters.
    """

    if not values or len(values) == 0:
        return None
    
    values = np.array(values)
    mean = np.mean(values)
    std_dev = np.std(values)

    new_values = values[np.abs(values - mean) < 3*std_dev]

    mean = np.mean(new_values)
    sigma = np.std(new_values)


    x_min = mean - 3*sigma
    x_max = mean + 3*sigma

    if not mc:
        if particle == "e":
            if energy < 20:
                n_bins = 25
            elif energy >= 20:
                n_bins = 17   
        
        if particle == "p":
            n_bins = 20
            if energy >= 90 and energy < 150:
                n_bins = 18
            elif energy >= 150 and energy < 200:
                n_bins = 16
            elif energy >= 200:
                n_bins = 14
            
        if particle == "c":
            n_bins = 15
            if energy >= 3000 and energy < 4500:
                if "m1" in title.lower() and energy >= 4000:
                    n_bins = 18
                    x_min = mean - 0.5*std_dev
                    x_max = mean + 0.5*std_dev
            elif energy >= 4500:
                if "m1" in title.lower():
                    n_bins = 25
                    x_min = mean - 0.2*std_dev
                    x_max = mean + 0.2*std_dev
    
    else:
        if particle == "e":
            if energy < 30:
                n_bins = 25
            elif energy >= 30 and energy < 80:
                n_bins = 20   
            elif energy >= 80:
                n_bins = 14
        
        if particle == "p":
            n_bins = 20
            if energy >= 60 and energy < 150:
                n_bins = 15
            elif energy >= 150 and energy < 200:
                n_bins = 10
            elif energy >= 200:
                n_bins = 10
            
        if particle == "c":
            n_bins = 15

    #n_bins = max(18, int(0.5*np.sqrt(len(new_values))))
    ##if n_bins > 40:
    ##    n_bins = 40
    #if energy < 20:
    #    n_bins = 12
    #else: 
    #    n_bins = 20
    
    '''
    if not mc:
        n_bins = 30
        if energy >= 30 and energy <= 122 and "M2" in title:
            n_bins -= 11
        if energy >= 50 and energy <= 122 and "M1" in title:
            n_bins -= 10
        if energy >= 122 and energy <= 180:
            n_bins -= 13
        if energy >= 180 and energy <= 3000:
            n_bins -= 15
        if energy >= 3000 and energy <= 4500:
            n_bins -= 18
        if energy > 4500 and "M1" in title:
            n_bins += 20
            x_min = mean - 0.4*std_dev
            x_max = mean + 0.4*std_dev
    else:
        n_bins = 30
        if energy >= 30 and energy <= 122 and "M2" in title:
            n_bins -= 11
        if energy >= 50 and energy <= 122 and "M1" in title:
            n_bins -= 11
        #if energy >= 122 and energy <= 180:
        #    n_bins -= 14
        if energy >= 122 and energy <= 150:
            n_bins -= 15
        if energy >= 150 and energy <= 4500:
            n_bins -= 18
        if energy > 4500:
            n_bins -= 18
            #x_min = mean - 0.4*std_dev
            #x_max = mean + 0.4*std_dev
            '''
    

    return {
        'mean': mean,
        'std_dev': std_dev,
        'x_min': x_min,
        'x_max': x_max,
        'n_bins': n_bins,
        'n_entries': len(new_values)
    }

# ============================================================
# Initialize and manage fit results structure
# ============================================================

def initialize_std_results():
    """Initialize the structure to store fit results (sigma and errors)"""
    return {
        "betap": [],
        "m1": {
            "l1": {"x_std_dev": [], "y_std_dev": [], "x_std_err": [], "y_std_err": []},
            "l2": {"x_std_dev": [], "y_std_dev": [], "x_std_err": [], "y_std_err": []},
            "l3": {"x_std_dev": [], "y_std_dev": [], "x_std_err": [], "y_std_err": []},
        },
        "m2": {
            "l1": {"x_std_dev": [], "y_std_dev": [], "x_std_err": [], "y_std_err": []},
            "l2": {"x_std_dev": [], "y_std_dev": [], "x_std_err": [], "y_std_err": []},
            "l3": {"x_std_dev": [], "y_std_dev": [], "x_std_err": [], "y_std_err": []},
        }
    }

def add_fit_results(std_results, betap, method, layer, coord, fitter):
    """
    Add fit results (sigma and error) to the results structure.
    
    Parameters:
    - std_results: results dictionary
    - betap: beta p value
    - method: 1 or 2
    - layer: 1, 2, or 3
    - coord: 'x' or 'y'
    - fitter: RooFitter object with completed fit
    """
    if fitter is None:
        return
    
    try:
        sigma = fitter.sigma_signal.getVal()
        sigma_err = fitter.sigma_signal.getError()
        
        layer_str = f"l{layer}"
        method_str = f"m{method}"
        coord_key_dev = f"{coord}_std_dev"
        coord_key_err = f"{coord}_std_err"
        
        std_results[method_str][layer_str][coord_key_dev].append(sigma)
        std_results[method_str][layer_str][coord_key_err].append(sigma_err)
    except Exception as e:
        print(f"⚠️ Error adding fit results: {e}")

# ============================================================

def load_all_trees_data(root_file_path):
    """
    Read ALL trees from the ROOT file.
    Each tree has: energy, betap, method, layer, res_x, res_y
    
    Returns a dictionary with structure:
    { tree_name: { 'energy': value, 'data': { betap: { method: { layer: {'x': [], 'y': []} } } } } }
    """
    f = ROOT.TFile(root_file_path, "READ")
    
    # Find all trees in the file
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
        
        # Structure for this tree
        tree_data = {
            'energy': None,  # Will be filled from first entry
            'data': {}
        }
        
        total_entries = 0
        
        # Loop through all events in the tree
        for entry in tree:
            energy = entry.energy
            betap = entry.betap
            method = entry.method
            layer = entry.layer
            res_x = entry.res_x
            res_y = entry.res_y
            
            # Save energy from first entry
            if tree_data['energy'] is None:
                tree_data['energy'] = energy
            
            # Nested structure
            tree_data['data'].setdefault(betap, {}).setdefault(method, {}).setdefault(layer, {'x': [], 'y': []})
            tree_data['data'][betap][method][layer]['x'].append(res_x)
            tree_data['data'][betap][method][layer]['y'].append(res_y)
            total_entries += 1
        
        all_data[tree_name] = tree_data
        
        # Statistics for this tree
        print(f"   Energy: {tree_data['energy']:.0f} MeV")
        print(f"   Total entries: {total_entries}")
        
        # Count per betap
        for betap in tree_data['data'].keys():
            n_events = sum(len(tree_data['data'][betap][method][layer]['x']) 
                           for method in tree_data['data'][betap] 
                           for layer in tree_data['data'][betap][method])
            print(f"   beta p={betap:.4f}: {n_events} entries")
    
    f.Close()
    return all_data

# ============================================================
# 2. Create histograms
# ============================================================

def create_histogram_adaptive(values, name, title, energy, mc, particle):
    """
    Create a TH1F from a list of values with adaptive x range and number of bins
    """
    
    if not values or len(values) == 0:
        return None
    
    #n_bins = 40
    #if energy >= 50 and "M2" in title:
    #    n_bins -= 10
    
    props = analyze_histogram_properties(values, energy, title, mc, particle)

    h = ROOT.TH1F(name, title, props["n_bins"], props["x_min"], props["x_max"])

    for v in values:
        h.Fill(v)
    
    print(f"  Histogram {name}:")
    print(f"    Mean: {props['mean']:.6f}, Std: {props['std_dev']:.6f}")
    print(f"    Range: [{props['x_min']:.6f}, {props['x_max']:.6f}]")
    print(f"    Bins: {props['n_bins']}, Entries: {props['n_entries']}")
    
    return h, props


# ============================================================
# Save histograms to ROOT file
# ============================================================

def save_histograms_to_root(all_histograms, output_file):
    """
    Save all histograms to a ROOT file for later inspection.
    
    Parameters:
    - all_histograms: dictionary with structure {tree_name: {betap: {method: {layer: {coord: hist}}}}}
    - output_file: path to output ROOT file
    """
    root_file = ROOT.TFile(output_file, "RECREATE")
    
    hist_count = 0
    
    for tree_name, tree_hists in all_histograms.items():
        # Create directory for this tree
        tree_dir = root_file.mkdir(f"tree_{tree_name}")
        tree_dir.cd()
        
        for betap, betap_hists in tree_hists.items():
            # Create directory for this betap
            betap_dir = tree_dir.mkdir(f"betap_{betap:.6f}")
            betap_dir.cd()
            
            for method, method_hists in betap_hists.items():
                # Create directory for this method
                method_dir = betap_dir.mkdir(f"method_{method}")
                method_dir.cd()
                
                for layer, layer_hists in method_hists.items():
                    # Create directory for this layer
                    layer_dir = method_dir.mkdir(f"layer_{layer}")
                    layer_dir.cd()
                    
                    for coord, hist in layer_hists.items():
                        if hist is not None:
                            hist.Write()
                            hist_count += 1
    
    root_file.Close()
    print(f"\n✅ Saved {hist_count} histograms to {output_file}")

# ============================================================
# 3. Fit with RooFitter
# ============================================================

def fit_residuals_adaptive(hist, props, coord_name, output_dir, suffix="", mc=False, method=1,):
    """
    Perform fit on histogram using RooFitter
    """

    if hist is None or hist.GetEntries() == 0:
        print(f"⚠️ Empty histogram for {suffix}, skipping fit")
        return None
    
    #total_entries = hist.GetEntries()
    
    fitter = RooFitter()
    fitter.hist = hist
    fitter.variable_range = [props['x_min'], props['x_max']]

    std_dev = props['std_dev']
    n_entries = props['n_entries']
    max_val = hist.GetMaximum()        

    #fitter.no_bkg = True
    
    # Set limits for the signal (peak around 0)
    #if method == 1 and "M1" in suffix:
    fitter.n_signal_limits = [max_val, max_val * 0.6, max_val * 1.4]
    fitter.mu_signal_limits = [0.0, -3 * std_dev, 3 * std_dev]
    if "8136" in suffix and method == 1 and not mc:
        fitter.sigma_signal_limits = [std_dev, std_dev * 0.015, std_dev * 1.5]
    elif "9.4" in suffix and method == 1:
        fitter.sigma_signal_limits = [std_dev, std_dev * 0.5, std_dev * 1.5]
    elif not mc:
        fitter.sigma_signal_limits = [std_dev, std_dev * 0.05, std_dev * 1.5]
    elif "8136" in suffix and method == 1 and mc:
        fitter.sigma_signal_limits = [std_dev, std_dev * 0.5, std_dev * 1.5]
    elif "4854" in suffix and method == 1 and mc:
        fitter.sigma_signal_limits = [std_dev, std_dev * 0.05, std_dev * 1.5]
    elif mc:
        fitter.sigma_signal_limits = [std_dev, std_dev * 0.03, std_dev * 1.5]
    if "2613" in suffix and method == 1 and not mc:
        fitter.alpha_left_signal_limits = [-2*std_dev, -3.0, -0.5]
        fitter.alpha_right_signal_limits = [2*std_dev, 0.5, 3.0]
        fitter.sigma_signal_limits = [std_dev, std_dev * 0.05, std_dev * 1.5]
    elif mc:
        fitter.alpha_left_signal_limits = [-2*std_dev, -3.0, -1.5]
        fitter.alpha_right_signal_limits = [2*std_dev, 1.5, 3.0]
    else:
        fitter.alpha_left_signal_limits = [-2*std_dev, -3.0, -0.8]
        fitter.alpha_right_signal_limits = [2*std_dev, 0.8, 3.0]  
        if "8136" in suffix and method == 1 and not mc:
            fitter.alpha_left_signal_limits = [-2*std_dev, -3.0, -1.0]
            fitter.alpha_right_signal_limits = [2*std_dev, 1.0, 3.0]
        elif "5554" in suffix and method == 1 and not mc:
            fitter.alpha_left_signal_limits = [-2*std_dev, -3.0, -0.7]
            fitter.alpha_right_signal_limits = [2*std_dev, 0.7, 3.0]

    fitter.integral_range = [-std_dev, std_dev]  # signal region
    
    # Exponential background (outside peak)
    fitter.exp_bkg = False
    #fitter.tau_background_limits = [-0.5, -10.0, -0.1]
    #fitter.n_background_limits = [n_entries * 0.3, max(10, n_entries * 0.05), n_entries * 1.5]
    
    fitter.no_bkg = True

    fitter.cent_label = coord_name.upper()
    fitter.pt_label = suffix
    
    fitter.initialise()
    
    try:
        fitter.fit()
        y_axis = fitter.frame.GetYaxis()
        y_axis.SetTitle("Entries")
        #fitter.saveFrameAsPDF(output_dir)
        return fitter
    except Exception as e:
        print(f"❌ Error in fit for {suffix}: {e}")
        return None

# ============================================================
# 4. Process a single tree (one energy)
# ============================================================

def process_single_tree(tree_name, tree_data, output_base_dir, mc, particle):
    """
    Process a single tree (one energy):
    1. Create histograms
    2. Perform fits
    3. Produce PDF with 2x2 layout
    """
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
    
    # Dictionary to store histograms for this tree
    tree_histograms = {}
    tree_fitters = {}
    legends = []
    
    # Initialize structure to store fit results for this tree
    std_results = initialize_std_results()
    
    # Collect all available betap values
    betap_values = sorted(data.keys())

    all_layers = set()
    for betap in betap_values:
        for method in data[betap].keys():
            all_layers.update(data[betap][method].keys())
    layers = sorted(all_layers)
    
    print(f"📊 Found layers: {layers}")
    
    # Create canvas for PDF
    canvas = ROOT.TCanvas(f"c_{tree_name}", f"Energy = {energy:.0f} MeV", 1200, 1000)
    
    first_page = True
    
    for betap in betap_values:
        tree_histograms[betap] = {1: {}, 2: {}}
        tree_fitters[betap] = {1: {}, 2: {}}
        
        # Add betap to results
        if betap not in std_results["betap"]:
            std_results["betap"].append(betap)
        
        for layer in layers:
            tree_histograms[betap][1][layer] = {}
            tree_histograms[betap][2][layer] = {}
            tree_fitters[betap][1][layer] = {}
            tree_fitters[betap][2][layer] = {}
            
            # Verify both methods exist for this layer
            if 1 not in data[betap] or 2 not in data[betap]:
                print(f"⚠️ beta={betap:.4f}: method 1 or 2 missing")
                continue
            if layer not in data[betap][1] or layer not in data[betap][2]:
                print(f"⚠️ beta={betap:.4f}, layer={layer}: missing data")
                continue
            
            # Create identifier for this combination
            suffix = f"betap{betap:.4f}_L{layer}"
            
            # Prepare 4 histograms
            h_x_m1, props_x_m1 = create_histogram_adaptive(
                data[betap][1][layer]['x'],
                f"h_x_m1_{suffix}",
                f"X residuals - M1, layer {layer}",
                energy,
                mc,
                particle
            )
            h_y_m1, props_y_m1 = create_histogram_adaptive(
                data[betap][1][layer]['y'],
                f"h_y_m1_{suffix}",
                f"Y residuals - M1, layer {layer}",
                energy,
                mc,
                particle
            )
            h_x_m2, props_x_m2 = create_histogram_adaptive(
                data[betap][2][layer]['x'],
                f"h_x_m2_{suffix}",
                f"X residuals - M2, layer {layer}",
                energy,
                mc,
                particle
            )
            h_y_m2, props_y_m2 = create_histogram_adaptive(
                data[betap][2][layer]['y'],
                f"h_y_m2_{suffix}",
                f"Y residuals - M2, layer {layer}",
                energy,
                mc,
                particle
            )
            
            # Store histograms for saving
            tree_histograms[betap][1][layer] = {'x': h_x_m1, 'y': h_y_m1}
            tree_histograms[betap][2][layer] = {'x': h_x_m2, 'y': h_y_m2}
            
            # Perform fits (signal only) and collect results
            if h_x_m1:
                fitter_x_m1 = fit_residuals_adaptive(h_x_m1, props_x_m1, 'X', output_base_dir, suffix, mc, method=1) 
                add_fit_results(std_results, betap, 1, layer, 'x', fitter_x_m1)
                print(f"    Fit results for beta={betap:.6f}, layer={layer}, method=1, coord=x: sigma={fitter_x_m1.sigma_signal.getVal():.6f} ± {fitter_x_m1.sigma_signal.getError():.6f} / RMS={h_x_m1.GetRMS():.6f} ± {h_x_m1.GetRMSError():.6f}")
            if h_y_m1:
                fitter_y_m1 = fit_residuals_adaptive(h_y_m1, props_y_m1, 'Y', output_base_dir, suffix, mc, method=1)
                add_fit_results(std_results, betap, 1, layer, 'y', fitter_y_m1)
                print(f"    Fit results for beta={betap:.6f}, layer={layer}, method=1, coord=y: sigma={fitter_y_m1.sigma_signal.getVal():.6f} ± {fitter_y_m1.sigma_signal.getError():.6f} / RMS={h_y_m1.GetRMS():.6f} ± {h_y_m1.GetRMSError():.6f}")
            if h_x_m2:
                fitter_x_m2 = fit_residuals_adaptive(h_x_m2, props_x_m2, 'X', output_base_dir, suffix, mc, method=2)
                add_fit_results(std_results, betap, 2, layer, 'x', fitter_x_m2)
                print(f"    Fit results for beta={betap:.6f}, layer={layer}, method=2, coord=x: sigma={fitter_x_m2.sigma_signal.getVal():.6f} ± {fitter_x_m2.sigma_signal.getError():.6f} / RMS={h_x_m2.GetRMS():.6f} ± {h_x_m2.GetRMSError():.6f}")
            if h_y_m2:
                fitter_y_m2 = fit_residuals_adaptive(h_y_m2, props_y_m2, 'Y', output_base_dir, suffix, mc, method=2)
                add_fit_results(std_results, betap, 2, layer, 'y', fitter_y_m2)
                print(f"    Fit results for beta={betap:.6f}, layer={layer}, method=2, coord=y: sigma={fitter_y_m2.sigma_signal.getVal():.6f} ± {fitter_y_m2.sigma_signal.getError():.6f} / RMS={h_y_m2.GetRMS():.6f} ± {h_y_m2.GetRMSError():.6f}")

            tree_fitters[betap][1][layer] = {'x': fitter_x_m1, 'y': fitter_y_m1}
            tree_fitters[betap][2][layer] = {'x': fitter_x_m2, 'y': fitter_y_m2}
            
            print(f"\n  📊 Creating 2x2 comparison for beta={betap:.6f}, layer={layer}")
            
            # Create 2x2 layout for summary PDF
            canvas.cd()
            canvas.Clear()
            canvas.Divide(2, 2)
            
            # Fill pads
            pads_config = [
                (1, 1, h_x_m1, fitter_x_m1, f'Residuals for X for M1 - #beta p {betap:.2f} MeV - Layer L{layer}'),
                (1, 2, h_y_m1, fitter_y_m1, f'Residuals for Y for M1 - #beta p {betap:.2f} MeV - Layer L{layer}'),
                (2, 1, h_x_m2, fitter_x_m2, f'Residuals for X for M2 - #beta p {betap:.2f} MeV - Layer L{layer}'),
                (2, 2, h_y_m2, fitter_y_m2, f'Residuals for Y for M2 - #beta p {betap:.2f} MeV - Layer L{layer}')
            ]
            
            for pad_row, pad_col, hist, fitter, title in pads_config:
                canvas.cd((pad_row-1)*2 + pad_col)
                #ROOT.gPad.SetGrid()
                ROOT.gPad.SetMargin(0.08, 0.04, 0.08, 0.06)
                
                if hist and hist.GetEntries() > 0 and fitter and fitter.frame:

                    frame_clone = fitter.frame.Clone()
                    frame_clone.SetTitle(title)
                    frame_clone.GetXaxis().SetTitle("Residual [mm]")
                    frame_clone.GetYaxis().SetTitle("Entries")

                    frame_clone.Draw()
                    
                    # Opzionale: aggiungi una legenda personalizzata
                    legend = ROOT.TLegend(0.55, 0.70, 0.92, 0.92)  # Posizione e dimensione
                    legend.SetBorderSize(0)
                    legend.SetFillStyle(0)
                    legend.SetTextSize(0.035)
                    
                    # Informazioni base
                    legend.AddEntry(hist, f"Entries: {int(hist.GetEntries())}", "f")
                    
                    # Parametri del fit dal fitter
                    if hasattr(fitter, 'mu_signal') and fitter.mu_signal:
                        mu_val = fitter.mu_signal.getVal()
                        mu_err = fitter.mu_signal.getError()
                        legend.AddEntry("", f"#mu = {mu_val:.4f} #pm {mu_err:.4f} mm", "")
                    
                    if hasattr(fitter, 'sigma_signal') and fitter.sigma_signal:
                        sigma_val = fitter.sigma_signal.getVal()
                        sigma_err = fitter.sigma_signal.getError()
                        legend.AddEntry("", f"#sigma = {sigma_val:.4f} #pm {sigma_err:.4f} mm", "")
                    
                    if hasattr(fitter, 'alpha_left_signal') and fitter.alpha_left_signal:
                        alphaL_val = fitter.alpha_left_signal.getVal()
                        alphaL_err = fitter.alpha_left_signal.getError()
                        legend.AddEntry("", f"#alpha_{{L}} = {alphaL_val:.3f} #pm {alphaL_err:.3f}", "")
                    
                    if hasattr(fitter, 'alpha_right_signal') and fitter.alpha_right_signal:
                        alphaR_val = fitter.alpha_right_signal.getVal()
                        alphaR_err = fitter.alpha_right_signal.getError()
                        legend.AddEntry("", f"#alpha_{{R}} = {alphaR_val:.3f} #pm {alphaR_err:.3f}", "")
                    
                    # Aggiungi chi-square se disponibile
                    if hasattr(fitter, 'chi2'):
                        legend.AddEntry(
                            "",
                            f"#chi^{{2}}/ndf = {fitter.chi2*fitter.ndf:.2f}/{fitter.ndf}",
                            ""
                        )
                    
                    print(f"    Creating legend for pad ({pad_row},{pad_col})")
                    legend.Draw()
                    print(f"    Legend drawn")
                    legends.append(legend) 

                    
                elif hist and hist.GetEntries() > 0:
                    # Se il fit è fallito, disegna solo l'istogramma
                    hist.SetTitle(title)
                    hist.GetXaxis().SetTitle("Residual [mm]")
                    hist.GetYaxis().SetTitle("Entries")
                    hist.SetLineColor(ROOT.kBlack)
                    hist.SetLineWidth(2)
                    hist.SetFillStyle(3001)
                    hist.SetFillColor(ROOT.kGray)
                    hist.Draw("E")
                    
                    # Legenda semplice
                    legend = ROOT.TLegend(0.65, 0.75, 0.9, 0.9)
                    legend.SetBorderSize(0)
                    legend.SetFillStyle(0)
                    legend.AddEntry(hist, f"Entries: {int(hist.GetEntries())}", "f")
                    legend.AddEntry("", "Fit failed", "")
                    legend.Draw()
                    
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
            
            # Page title
            canvas.cd(0)
            #title_obj = ROOT.TLatex()
            #title_obj.SetTextAlign(22)
            #title_obj.SetTextSize(0.03)
            #title_obj.DrawLatex(0.5, 0.97, f"Tree: {tree_name} | Energy = {energy:.0f} MeV | #beta p = {betap:.6f}")
            
            canvas.Update()
            
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

# ============================================================
# 5. Main function that processes ALL trees
# ============================================================

def process_all_trees(input_file, output_dir, particle, mc):
    """
    Process ALL trees in the ROOT file:
    1. Read all trees
    2. For each tree, create histograms, perform fits and produce PDF
    3. Save all histograms to ROOT file
    """
    # Load all trees
    print(f"\n📖 Reading file: {input_file}")
    all_data = load_all_trees_data(input_file)
    
    if not all_data:
        print(f"❌ Error: No trees found in file {input_file}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect histograms from all trees
    all_histograms = {}
    
    # Collect all fit results
    all_std_results = {
        "betap": [],
        "m1": {
            "l1": {"x_std_dev": [], "y_std_dev": [], "x_std_err": [], "y_std_err": []},
            "l2": {"x_std_dev": [], "y_std_dev": [], "x_std_err": [], "y_std_err": []},
            "l3": {"x_std_dev": [], "y_std_dev": [], "x_std_err": [], "y_std_err": []},
        },
        "m2": {
            "l1": {"x_std_dev": [], "y_std_dev": [], "x_std_err": [], "y_std_err": []},
            "l2": {"x_std_dev": [], "y_std_dev": [], "x_std_err": [], "y_std_err": []},
            "l3": {"x_std_dev": [], "y_std_dev": [], "x_std_err": [], "y_std_err": []},
        }
    }
    
    # Process each tree (each energy)
    print(f"\n{'='*60}")
    print(f"🎯 Processing {len(all_data)} trees found")
    print(f"{'='*60}")
    
    for tree_name, tree_data in all_data.items():
        tree_histograms, std_results = process_single_tree(tree_name, tree_data, output_dir, mc, particle)
        all_histograms[tree_name] = tree_histograms
        
        # Merge results from this tree
        for method in ["m1", "m2"]:
            for layer in ["l1", "l2", "l3"]:
                for coord in ["x", "y"]:
                    all_std_results[method][layer][f"{coord}_std_dev"].extend(
                        std_results[method][layer][f"{coord}_std_dev"]
                    )
                    all_std_results[method][layer][f"{coord}_std_err"].extend(
                        std_results[method][layer][f"{coord}_std_err"]
                    )
        
        # Ensure betap list is consistent (should be same for all trees)
        for betap in std_results["betap"]:
            if betap not in all_std_results["betap"]:
                all_std_results["betap"].append(betap)

    base = os.path.basename(input_file)

    prefix = base.split("residuals")[0].rstrip("_")
    
    # Save all histograms to ROOT file
    #histograms_output = os.path.join(output_dir, f"{prefix}_residuals_histograms.root")
    #save_histograms_to_root(all_histograms, histograms_output)

    if particle == "e":
        part = "electron"
    elif particle == "p":
        part = "proton"
    elif particle == "c":
        part = "carbon"
    
    # Save all fit results to JSON
    json_dir = "/home/lattina/limadou/data/testBeam/paper_plot_json/new_method"
    results_output = os.path.join(json_dir, f"{prefix}_residuals_results_roofit.json")
    with open(results_output, 'w') as f:
        fin_data = {
            "particle": part,
            "results": all_std_results
        }     

        json.dump(fin_data, f, indent=4)
    print(f"✅ Fit results saved to: {results_output}")
    
# ============================================================
# 6. MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit residuals distributions from ROOT file with multiple trees")
    parser.add_argument("--input-file", required=True, 
                        help="Input ROOT file containing multiple residuals trees")
    parser.add_argument("--output-dir", required=True, 
                        help="Output directory for fit results and PDFs")
    parser.add_argument("--particle", required=True, help="Select particle configuration.")
    parser.add_argument("--mc", action="store_true", help="Define name.")


    args = parser.parse_args()
    
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