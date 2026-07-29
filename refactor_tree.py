import uproot
import awkward as ak
import numpy as np
import argparse
import os
import ROOT
import array
from efficiency_utils_new import compute_gen_mask

def merge_mc(f_in, output_file, num):
    
    n = int(num)
    total_groups = 0
    
    # Crea il file di output
    f_out = ROOT.TFile(output_file, "UPDATE")
    
    # 2. Processa MCtruth e crea il nuovo tree
    #print("\n📊 Processing MCtruth...")
    mc_tree = f_out.Get("MCtruth")
    l1_tree = f_in.Get("L1")
    
    if not mc_tree:
        raise RuntimeError("MCtruth Tree Not found")
    if not l1_tree:
        raise RuntimeError("L1 Tree Not found")
    
    n_entries = mc_tree.GetEntries()
    print(f"Processing {n_entries} events, merging every {n}")

    # =========================
    # ===== MCtruth OUTPUT ====
    # =========================

    new_mc = ROOT.TTree("MCtruth", "merged MCtruth") 
    
    # Definisci i branch per l'output
    # Vettori per tracce
    particle_id = ROOT.vector('int')()
    event_index = ROOT.vector('int')()
    gen_theta = ROOT.vector('double')()
    gen_phi = ROOT.vector('double')()
    gen_energy = ROOT.vector('double')()
    mass = ROOT.vector('double')()
    
    # Vettori per cluster
    cluster_particle_id = ROOT.vector('int')()
    x_abspos = ROOT.vector('double')()
    y_abspos = ROOT.vector('double')()
    cl_layer = ROOT.vector('int')()
    csize = ROOT.vector('int')()
    
    # Vettori per rivelatori
    tr1_posX = ROOT.vector('double')()
    tr1_posY = ROOT.vector('double')()
    tr2_posX = ROOT.vector('double')()
    tr2_posY = ROOT.vector('double')()

    cls_to_trk_idx = ROOT.vector('int')()
    
    # Crea i branch
    new_mc.Branch("particle_id", particle_id)
    new_mc.Branch("gen_theta", gen_theta)
    new_mc.Branch("gen_phi", gen_phi)
    new_mc.Branch("gen_energy", gen_energy)
    new_mc.Branch("Mass", mass)
    new_mc.Branch("cluster_particle_id", cluster_particle_id)
    new_mc.Branch("x_abspos", x_abspos)
    new_mc.Branch("y_abspos", y_abspos)
    new_mc.Branch("cl_layer", cl_layer)
    new_mc.Branch("csize", csize)
    new_mc.Branch("TR1_posX", tr1_posX)
    new_mc.Branch("TR1_posY", tr1_posY)
    new_mc.Branch("TR2_posX", tr2_posX)
    new_mc.Branch("TR2_posY", tr2_posY)
    new_mc.Branch("event_index", event_index)
    new_mc.Branch("cls_to_trk_idx", cls_to_trk_idx)


    # =========================
    # ===== L1 OUTPUT =========
    # =========================

    new_l1 = ROOT.TTree("L1", "merged L1")

    out_pix_col = ROOT.vector('int')()
    out_pix_row = ROOT.vector('int')()
    out_chip_id = ROOT.vector('int')()
    out_event_index = ROOT.vector('int')()

    new_l1.Branch("DIR_pix_col", out_pix_col)
    new_l1.Branch("DIR_pix_row", out_pix_row)
    new_l1.Branch("DIR_chip_id", out_chip_id)
    new_l1.Branch("event_index", out_event_index)

    boot_nr = array.array('h', [0])
    run_nr = array.array('h', [0])

    TR1_countHG = array.array('d', [0.0]*10)
    TR2_countHG = array.array('d', [0.0]*8)
    RAN_countHG = array.array('d', [0.0]*24)
    EN1_countHG = array.array('d', [0.0]*6)
    EN2_countHG = array.array('d', [0.0]*6)
    VETO_countHG = array.array('d', [0.0]*10)

    TR1_countLG = array.array('d', [0.0]*10)
    TR2_countLG = array.array('d', [0.0]*8)
    RAN_countLG = array.array('d', [0.0]*24)
    EN1_countLG = array.array('d', [0.0]*6)
    EN2_countLG = array.array('d', [0.0]*6)
    VETO_countLG = array.array('d', [0.0]*10)

    new_l1.Branch("boot_nr", boot_nr, "boot_nr/s")
    new_l1.Branch("run_nr", run_nr, "run_nr/s")
    new_l1.Branch("TR1_countHG[5][2]", TR1_countHG, "TR1_countHG[5][2]/D")
    new_l1.Branch("TR2_countHG[4][2]", TR2_countHG, "TR2_countHG[4][2]/D")
    new_l1.Branch("RAN_countHG[12][2]", RAN_countHG, "RAN_countHG[12][2]/D")
    new_l1.Branch("EN1_countHG[3][2]", EN1_countHG, "EN1_countHG[3][2]/D")
    new_l1.Branch("EN2_countHG[3][2]", EN2_countHG, "EN2_countHG[3][2]/D")
    new_l1.Branch("VETO_countHG[5][2]", VETO_countHG, "VETO_countHG[5][2]/D")

    new_l1.Branch("TR1_countLG[5][2]", TR1_countLG, "TR1_countLG[5][2]/D")
    new_l1.Branch("TR2_countLG[4][2]", TR2_countLG, "TR2_countLG[4][2]/D")
    new_l1.Branch("RAN_countLG[12][2]", RAN_countLG, "RAN_countLG[12][2]/D")
    new_l1.Branch("EN1_countLG[3][2]", EN1_countLG, "EN1_countLG[3][2]/D")
    new_l1.Branch("EN2_countLG[3][2]", EN2_countLG, "EN2_countLG[3][2]/D")
    new_l1.Branch("VETO_countLG[5][2]", VETO_countLG, "VETO_countLG[5][2]/D")


    # =========================
    # ===== INPUT BRANCHES ====
    # =========================

    # MC
    gen_theta_orig = ROOT.vector('double')()
    gen_phi_orig = ROOT.vector('double')()
    x_abspos_orig = ROOT.vector('double')()
    y_abspos_orig = ROOT.vector('double')()
    cl_layer_orig = ROOT.vector('int')()
    csize_orig = ROOT.vector('int')()
    tr1_posX_orig = ROOT.vector('double')()
    tr1_posY_orig = ROOT.vector('double')()
    tr2_posX_orig = ROOT.vector('double')()
    tr2_posY_orig = ROOT.vector('double')()
    cls_to_trk_idx_orig = ROOT.vector('int')()
    event_index_orig = array.array('i', [0])  # int singolo
    gen_energy_orig = ROOT.vector('double')()
    mass_orig = ROOT.vector('double')()


    mc_tree.SetBranchAddress("gen_theta", gen_theta_orig)
    mc_tree.SetBranchAddress("gen_phi", gen_phi_orig)
    mc_tree.SetBranchAddress("x_abspos", x_abspos_orig)
    mc_tree.SetBranchAddress("y_abspos", y_abspos_orig)
    mc_tree.SetBranchAddress("cl_layer", cl_layer_orig)
    mc_tree.SetBranchAddress("csize", csize_orig)
    mc_tree.SetBranchAddress("TR1_posX", tr1_posX_orig)
    mc_tree.SetBranchAddress("TR1_posY", tr1_posY_orig)
    mc_tree.SetBranchAddress("TR2_posX", tr2_posX_orig)
    mc_tree.SetBranchAddress("TR2_posY", tr2_posY_orig)
    mc_tree.SetBranchAddress("cls_to_trk_idx", cls_to_trk_idx_orig)
    mc_tree.SetBranchAddress("event_index", event_index_orig)
    mc_tree.SetBranchAddress("gen_energy", gen_energy_orig)
    mc_tree.SetBranchAddress("Mass", mass_orig)



    # L1
    dir_pix_col = ROOT.vector('int')()
    dir_pix_row = ROOT.vector('int')()
    dir_chip_id = ROOT.vector('int')()

    '''
    boot_nr_in = array.array('h', [0])
    run_nr_in = array.array('h', [0])

    TR1_countHG_in = array.array('d', [0.0]*10)
    TR2_countHG_in = array.array('d', [0.0]*8)
    RAN_countHG_in = array.array('d', [0.0]*24)
    EN1_countHG_in = array.array('d', [0.0]*6)
    EN2_countHG_in = array.array('d', [0.0]*6)
    VETO_countHG_in = array.array('d', [0.0]*10)

    TR1_countLG_in = array.array('d', [0.0]*10)
    TR2_countLG_in = array.array('d', [0.0]*8)
    RAN_countLG_in = array.array('d', [0.0]*24)
    EN1_countLG_in = array.array('d', [0.0]*6)
    EN2_countLG_in = array.array('d', [0.0]*6)
    VETO_countLG_in = array.array('d', [0.0]*10)
    '''

    l1_tree.SetBranchAddress("DIR_pix_col", dir_pix_col)
    l1_tree.SetBranchAddress("DIR_pix_row", dir_pix_row)
    l1_tree.SetBranchAddress("DIR_chip_id", dir_chip_id)

    '''
    l1_tree.SetBranchAddress("boot_nr", boot_nr_in)
    l1_tree.SetBranchAddress("run_nr", run_nr_in)

    l1_tree.SetBranchAddress("TR1_countHG", TR1_countHG_in)
    l1_tree.SetBranchAddress("TR2_countHG", TR2_countHG_in)
    l1_tree.SetBranchAddress("RAN_countHG", RAN_countHG_in)
    l1_tree.SetBranchAddress("EN1_countHG", EN1_countHG_in)
    l1_tree.SetBranchAddress("EN2_countHG", EN2_countHG_in)
    l1_tree.SetBranchAddress("VETO_countHG", VETO_countHG_in)

    l1_tree.SetBranchAddress("TR1_countLG", TR1_countLG_in)
    l1_tree.SetBranchAddress("TR2_countLG", TR2_countLG_in)
    l1_tree.SetBranchAddress("RAN_countLG", RAN_countLG_in)
    l1_tree.SetBranchAddress("EN1_countLG", EN1_countLG_in)
    l1_tree.SetBranchAddress("EN2_countLG", EN2_countLG_in)
    l1_tree.SetBranchAddress("VETO_countLG", VETO_countLG_in)
    '''

    for i in range(0, n_entries - n + 1, n):
        #print(f"\nProcessing {i} → {i+n-1}")

        # reset MC
        particle_id.clear()
        event_index.clear()
        gen_theta.clear()
        gen_phi.clear()
        gen_energy.clear()
        mass.clear()
        cluster_particle_id.clear()
        x_abspos.clear()
        y_abspos.clear()
        cl_layer.clear()
        csize.clear()
        tr1_posX.clear()
        tr1_posY.clear()
        tr2_posX.clear()
        tr2_posY.clear()
        cls_to_trk_idx.clear()

        # reset L1
        out_pix_col.clear()
        out_pix_row.clear()
        out_chip_id.clear()
        out_event_index.clear()

        for idx in range(len(TR1_countHG)):
            TR1_countHG[idx] = TR1_countLG[idx] = 0
            VETO_countHG[idx] = VETO_countLG[idx] = 0

        for idx in range(len(TR2_countHG)):
            TR2_countHG[idx] = TR2_countLG[idx] = 0

        for idx in range(len(RAN_countHG)):
            RAN_countHG[idx] = RAN_countLG[idx] = 0

        for idx in range(len(EN1_countHG)):
            EN1_countHG[idx] = EN1_countLG[idx] = 0
            EN2_countHG[idx] = EN2_countLG[idx] = 0

        clusters_per_event = []

        for j in range(n):
            entry = i + j

            # ===== MC =====
            mc_tree.GetEntry(entry)

            particle_id.push_back(j)
            event_index.push_back(event_index_orig[0])

            gen_theta.push_back(gen_theta_orig[0])
            gen_phi.push_back(gen_phi_orig[0])
            gen_energy.push_back(gen_energy_orig[0])
            mass.push_back(mass_orig[0])

            n_cl = x_abspos_orig.size()
            clusters_per_event.append(n_cl)
 
            for k in range(n_cl):
                x_abspos.push_back(x_abspos_orig[k])
                y_abspos.push_back(y_abspos_orig[k])
                cl_layer.push_back(cl_layer_orig[k])
                csize.push_back(csize_orig[k])

            for k in range(tr1_posX_orig.size()):
                tr1_posX.push_back(tr1_posX_orig[k])
                tr1_posY.push_back(tr1_posY_orig[k])

            for k in range(tr2_posX_orig.size()):
                tr2_posX.push_back(tr2_posX_orig[k])
                tr2_posY.push_back(tr2_posY_orig[k])

            for k in range(cls_to_trk_idx_orig.size()):
                if cls_to_trk_idx_orig[k] == -999:
                    cls_to_trk_idx.push_back(-999)
                else:
                    cls_to_trk_idx.push_back(j)


            # ===== L1 =====
            if entry < l1_tree.GetEntries():
                l1_tree.GetEntry(event_index_orig[0])

                for v in dir_pix_col:
                    out_pix_col.push_back(v)
                for v in dir_pix_row:
                    out_pix_row.push_back(v)
                for v in dir_chip_id:
                    out_chip_id.push_back(v)

                out_event_index.push_back(event_index_orig[0])

            if j == 0:
                boot_nr[0] = 0
                run_nr[0] = 0

        for ev_idx, n_cl in enumerate(clusters_per_event):
            for _ in range(n_cl):
                cluster_particle_id.push_back(ev_idx)

        new_mc.Fill()
        new_l1.Fill()
        total_groups += 1

    f_out.Write()

    # Trova il ciclo più alto (l'ultimo salvato)
    max_cycle = 0
    keys = f_out.GetListOfKeys()
    for key in keys:
        if key.GetName() == "MCtruth":
            cycle = key.GetCycle()
            if cycle > max_cycle:
                max_cycle = cycle

    #print(f"Ultimo ciclo trovato: MCtruth;{max_cycle}")

    # Elimina tutti i cicli tranne l'ultimo
    for key in keys:
        if key.GetName() == "MCtruth":
            cycle = key.GetCycle()
            if cycle != max_cycle:
                f_out.Delete(f"MCtruth;{cycle}")
                #print(f"Eliminato MCtruth;{cycle}")

    # Rinomina l'ultimo ciclo in MCtruth (senza numero)
    #mc_rename = f_out.Get(f"MCtruth;{max_cycle}")
    #f_out.Rename(f"MCtruth;{max_cycle}", "MCtruth")
    #print(f"Rinominato MCtruth;{max_cycle} in MCtruth")

    f_out.Purge()
    #f_out.Write()


    f_out.Close()

    print(f"\n" + "="*50)
    print(f"EXECUTION COMPLETED!")
    print(f"="*50)
    #print(f"MCtruth starting events: {mc_tree.GetEntries()}")
    #print(f"Valid events: {valid_events_count}")
    #print(f"Skipped events: {skipped_events_count}")
    print(f"Groups created (each group = {n} events): {total_groups}")
    print(f"Discarded events: {n_entries % n}")
    print(f"Output file saved: {output_file}")
    print("✅ DONE: MCtruth + L1 merged correttamente")


            
def mctruth_gen_mask(output_file):    

    # 2. Processa MCtruth
    print("\n📂 Processing MCtruth tree...")
    
    with uproot.open(output_file) as f:
        mc_tree_uproot = f["MCtruth"]
        #gen_mask = np.ones(mc_tree_uproot.num_entries, dtype=bool)        
        gen_mask = compute_gen_mask(mc_tree_uproot, [], verbose = True)  # Usa la tua funzione esistente
        #gen_mask = 
    
    total_events = len(gen_mask)
    selected_events = np.sum(gen_mask)
    
    print(f"Total events: {total_events}")
    print(f"Selected events: {selected_events}")
    
    if selected_events == 0:
        print("⚠️ Warning: No events selected!")
        output_file.Close()
        return
    
    # Ora copia gli eventi selezionati usando ROOT
    
    f_out = ROOT.TFile(output_file, "UPDATE")  # Apri in modalità UPDATE per modificare il file esistente
    original_tree = f_out.Get("MCtruth")
    new_tree = original_tree.CloneTree(0)  # Clona la struttura senza eventi

    # Aggiungi il branch event_index
    event_index = np.zeros(1, dtype=np.int32)
    new_tree.Branch("event_index", event_index, "event_index/I")
    
    # Copia solo gli eventi selezionati
    print("\n📝 Copying selected events...")
    for i in range(total_events):
        if gen_mask[i]:
            original_tree.GetEntry(i)
            event_index[0] = i  # Salva la posizione originale
            new_tree.Fill()
    
    f_out.cd()
    new_tree.Write("MCtruth", ROOT.TObject.kOverwrite)
    print(f"✅ MCtruth tree written with {new_tree.GetEntries()} events")

    f_out.Close()



def match_hit_to_cls(input_file, output_dir="../data/mc_sim/L1/", num=1):

    PIX_PITCH = 0.03 # in mm

    print(f"\nProcessing file: {input_file}")

    f_in = ROOT.TFile(input_file, "READ")
    tree_in = f_in.Get("MCtruth")

    filename = os.path.basename(input_file) 
    filename = os.path.splitext(filename)[0]

    output_file = os.path.join(output_dir, f"{filename}_{num}_evs.root")
    f_out = ROOT.TFile(output_file, "RECREATE")
    tree_out = ROOT.TTree("MCtruth", "MCtruth with clusters - tracks matching")

    print(f"\nEvent number in input file: {tree_in.GetEntries()}")

    branches = [
        "gen_theta", "gen_phi", "Mass", "gen_energy",
        "Hit_PosX", "Hit_PosY", "Hit_PosZ",
        "Hit_TrackId", "Hit_ParentId",
        "csize", "x_abspos", "y_abspos",
        "cl_layer", "TR1_posX", "TR1_posY", "TR2_posX", "TR2_posY"
    ]

    buffers = {}

    for b in branches:
        # Prendi il tipo del branch
        leaf = tree_in.GetLeaf(b)
        leaf_type = leaf.GetTypeName()
        #print(f"Branch: {b}, Type: {leaf_type}")

        if "vector" in leaf_type:
            # Se è un vettore di tipo float/int -> usare ROOT.std.vector
            if "double" in leaf_type:
                buffers[b] = ROOT.vector('double')()
            else:
                buffers[b] = ROOT.vector('int')()
            tree_out.Branch(b, buffers[b])
        else:
            # Se è singolo valore
            if "Double_t" in leaf_type or "double" in leaf_type:
                buffers[b] = array('d', [0.0])
                tree_out.Branch(b, buffers[b], f"{b}/D")
            elif "Int_t" in leaf_type or "int" in leaf_type:
                buffers[b] = array('i', [0])
                tree_out.Branch(b, buffers[b], f"{b}/I")
            else:
                buffers[b] = array('h', [0])  # short int
                tree_out.Branch(b, buffers[b], f"{b}/S")

    cls_buffer = ROOT.vector('int')() 
    tree_out.Branch("cls_to_trk_idx", cls_buffer)

    n_events = tree_in.GetEntries()

    for evt in range(n_events):
        tree_in.GetEntry(evt)

        cls_buffer.clear()

        for b in branches:
            buffers[b].clear() 
            leaf = getattr(tree_in, b)
            for val in leaf:
                buffers[b].push_back(val)
                #print(f"  {b} -> {val}")  # mostra tutti i valori per questo evento

        n_clusters = len(buffers["csize"])
        cls_to_track = [-999] * n_clusters

        for par_id_pos in range(len(buffers["Hit_ParentId"])):
            if buffers["Hit_ParentId"][par_id_pos] != 0:
                continue
            if buffers["Hit_TrackId"][par_id_pos] != 1:
                continue
            
            hit_x = buffers["Hit_PosX"][par_id_pos]
            hit_y = buffers["Hit_PosY"][par_id_pos]
            hit_z = buffers["Hit_PosZ"][par_id_pos]

            for cls_n in range(n_clusters):
                r = buffers["csize"][cls_n] * PIX_PITCH
                dx = buffers["x_abspos"][cls_n] - hit_x
                dy = buffers["y_abspos"][cls_n] - hit_y
                d = np.sqrt(dx*dx + dy*dy)

                if abs(hit_z - 17.825) < 2:
                    hit_layer = 0
                elif abs(hit_z - 26.325) < 2:
                    hit_layer = 1
                elif abs(hit_z - 34.825) < 2:
                    hit_layer = 2
                else:
                    hit_layer = -1

                if hit_layer != -1 and d < r and hit_layer == buffers["cl_layer"][cls_n]:
                    cls_to_track[cls_n] = buffers["Hit_TrackId"][par_id_pos]
    
        # Riempie il vector ROOT
        cls_buffer.clear()
        for val in cls_to_track:
            cls_buffer.push_back(val)
        
        #print(f"Evento {evt}: cls_to_trk_idx -> {list(cls_buffer)}")

        tree_out.Fill() 
    
    tree_out.Write()  # Scrive il TTree nel file di output

    f_out.Write()          # Scrive tutti gli oggetti nel file
    f_out.Close()          # Chiude il file di output

    #print(f"\n✅ MCtruth tree with matched clusters-tracks written with {tree_out.GetEntries()} events")

    mctruth_gen_mask(output_file)

    merge_mc(f_in, output_file, num)

    f_out = ROOT.TFile(output_file, "UPDATE")  # Apri in modalità UPDATE per modificare il file esistente

    tree_ls = ["MCdig", "MCcls", "Tmd", "TCalib"]
    
    print("\n📂 Copying TTrees from input file...")
    for tree_name in tree_ls:
        tree = f_in.Get(tree_name)
        if tree and tree.InheritsFrom("TTree"):
            # Clona il tree
            cloned_tree = tree.CloneTree(-1, "fast")
            cloned_tree.Write()
            print(f"✅ {tree_name} tree copied to output ROOT file.")
        else:
            print(f"⚠️ Warning: {tree_name} not found or not a TTree")

    f_out.Write()          # Scrive tutti gli oggetti nel file
    f_out.Close()          # Chiude il file di output
    f_in.Close()          # Chiude il file di input



# Esempio di utilizzo
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge n MC tracks (useful for efficiency studies).")
    parser.add_argument("--input-file", required=True, help="Extract data from MCtruth from this L1.root file.")
    parser.add_argument("--output-dir", default="./output", help="Define output directory.")
    parser.add_argument("--num", type=int, default=1, help="Number of MC tracks to merge.")
    args = parser.parse_args()
       
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    match_hit_to_cls(args.input_file, output_dir, args.num)
    #mctruth_gen_mask(args.input_file, output_dir)

  