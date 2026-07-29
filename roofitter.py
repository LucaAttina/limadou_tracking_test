import ROOT
import numpy as np

import sys

sys.path.append("utils/roofit_functions")

ROOT.ROOT.EnableImplicitMT(20)
ROOT.RooMsgService.instance().setSilentMode(True)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

# load custom PDFs
ROOT.gROOT.LoadMacro(r"utils/roofit_functions/RooGausDExp.cxx++")
ROOT.gROOT.LoadMacro(r"utils/roofit_functions/RooGausExp.cxx++")


class RooFitter:
    def __init__(self):
        self.variable = None
        self.variable_range = []

        self.hist = None

        self.exp_bkg = False

        # signal
        self.n_signal = None
        self.n_signal_limits = [1e2, 1e-4, 1e4]
        self.mu_signal = None
        self.mu_signal_limits = [0, -1.5, 2]
        self.sigma_signal = None
        self.sigma_signal_limits = [0.1, 0.01, 1.5]
        self.alpha_left_signal = None
        self.alpha_left_signal_limits = [-2.0, -4.0, -0.5]
        self.alpha_right_signal = None
        self.alpha_right_signal_limits = [2.0, 0.5, 4.0]

        self.model_signal = None

        # background
        self.no_bkg = False
        self.n_background = None
        self.n_background_limits = [1e2, 1e-3, 1e4]

        # gaussian background
        self.mu_background = None
        self.mu_background_limits = [-4.0, -5.0, -2.0]
        self.sigma_background = None
        self.sigma_background_limits = [1.0, 0.5, 1.5]
        self.alpha_left_background = None
        self.alpha_left_background_limits = [-2.0, -4.0, -0.5]
        self.alpha_right_background = None
        self.alpha_right_background_limits = [2.0, 0.5, 4.0]

        # exponential background
        self.tau_background = None
        self.tau_background_limits = [-0.5, -10.0, -0.2]

        self.model_background = None

        # total
        self.total_model = None

        self.fit_results = None
        self.tot_integral = 0.0
        self.tot_integral_error = 0.0
        self.bkg_integral = 0.0
        self.bkg_integral_error = 0.0
        self.purity = 1.0
        self.purity_error = 0.0

        # integral range
        self.integral_range = []

        self.frame = None
        self.info_panel = None
        self.canvas = None

        self.pt_label = ""
        self.cent_label = ""

    def _check_members(self):
        if self.hist is None:
            raise ValueError(f"histogram not set")
        if self.variable is None:
            raise ValueError(f"variable not correctly set")
        if not self.variable_range:
            raise ValueError(f"variable range is empty")

        if self.n_signal is None:
            raise ValueError(f"n_signal not correctly set")
        if self.mu_signal is None:
            raise ValueError(f"mu_signal not correctly set")
        if self.sigma_signal is None:
            raise ValueError(f"sigma_signal not correctly set")
        if self.alpha_left_signal is None:
            raise ValueError(f"alpha_left_signal not correctly set")
        if self.alpha_right_signal is None:
            raise ValueError(f"alpha_right_signal not correctly set")
        if self.model_signal is None:
            raise ValueError(f"model_signal not correctly set")

        if self.n_background is None:
            raise ValueError(f"n_background not correctly set")

        if self.exp_bkg:
            # exponential background
            if self.tau_background is None:
                raise ValueError(f"tau_background_background not correctly set")
        else:
            if self.mu_background is None:
                raise ValueError(f"mu_background not correctly set")
            if self.sigma_background is None:
                raise ValueError(f"sigma_background not correctly set")
            if self.alpha_left_background is None:
                raise ValueError(f"alpha_left_background not correctly set")
            if self.alpha_right_background is None:
                raise ValueError(f"alpha_right_background not correctly set")

        if self.model_background is None:
            raise ValueError(f"model_background_background not correctly set")

        if self.total_model is None:
            raise ValueError(f"total_model not correctly set")

        if not self.integral_range:
            raise ValueError(f"integral_range is empty")

    def initialise(self):

        self.variable = ROOT.RooRealVar(
            r"Residuals",
            r"Residuals",
            self.variable_range[0],
            self.variable_range[1],
            "mm",
        )

        # signal
        self.n_signal = ROOT.RooRealVar(
            r"Entries",
            r"Entries",
            self.n_signal_limits[0],
            self.n_signal_limits[1],
            self.n_signal_limits[2],
        )
        self.mu_signal = ROOT.RooRealVar(
            r"#mu_{sig}",
            r"#mu_{sig}",
            self.mu_signal_limits[0],
            self.mu_signal_limits[1],
            self.mu_signal_limits[2],
        )
        self.sigma_signal = ROOT.RooRealVar(
            r"#sigma_{sig}",
            r"#sigma_{sig}",
            self.sigma_signal_limits[0],
            self.sigma_signal_limits[1],
            self.sigma_signal_limits[2],
        )
        self.alpha_left_signal = ROOT.RooRealVar(
            r"#alpha_{L, sig}",
            r"#alpha_{L, sig}",
            self.alpha_left_signal_limits[0],
            self.alpha_left_signal_limits[1],
            self.alpha_left_signal_limits[2],
        )
        self.alpha_right_signal = ROOT.RooRealVar(
            r"#alpha_{R, sig}",
            r"#alpha_{R, sig}",
            self.alpha_right_signal_limits[0],
            self.alpha_right_signal_limits[1],
            self.alpha_right_signal_limits[2],
        )

        self.model_signal = ROOT.RooGausDExp(
            "signal",
            "signal",
            self.variable,
            self.mu_signal,
            self.sigma_signal,
            self.alpha_left_signal,
            self.alpha_right_signal,
        )

        if not self.n_background:

            # background
            self.n_background = ROOT.RooRealVar(
                r"N_{bkg}",
                r"N_{bkg}",
                self.n_background_limits[0],
                self.n_background_limits[1],
                self.n_background_limits[2],
            )
            if self.exp_bkg:
                # exponential background
                self.tau_background = ROOT.RooRealVar(
                    r"#tau_{bkg}",
                    r"#tau_{bkg}",
                    self.tau_background_limits[0],
                    self.tau_background_limits[1],
                    self.tau_background_limits[2],
                )

                self.model_background = ROOT.RooExponential(
                    "background", "background", self.variable, self.tau_background
                )
            else:
                self.mu_background = ROOT.RooRealVar(
                    r"#mu_{bkg}",
                    r"#mu_{bkg}",
                    self.mu_background_limits[0],
                    self.mu_background_limits[1],
                    self.mu_background_limits[2],
                )
                self.sigma_background = ROOT.RooRealVar(
                    r"#sigma_{bkg}",
                    r"#sigma_{bkg}",
                    self.sigma_background_limits[0],
                    self.sigma_background_limits[1],
                    self.sigma_background_limits[2],
                )
                self.alpha_left_background = ROOT.RooRealVar(
                    r"#alpha_{L, sig}",
                    r"#alpha_{L, sig}",
                    self.alpha_left_background_limits[0],
                    self.alpha_left_background_limits[1],
                    self.alpha_left_background_limits[2],
                )
                self.alpha_right_background = ROOT.RooRealVar(
                    r"#alpha_{R, sig}",
                    r"#alpha_{R, sig}",
                    self.alpha_right_background_limits[0],
                    self.alpha_right_background_limits[1],
                    self.alpha_right_background_limits[2],
                )

                self.model_background = ROOT.RooGausDExp(
                    "background",
                    "background",
                    self.variable,
                    self.mu_background,
                    self.sigma_background,
                    self.alpha_left_background,
                    self.alpha_right_background,
                )

        # total model
        if self.no_bkg:
            self.total_model = ROOT.RooAddPdf(
                "model",
                "total model",
                ROOT.RooArgList(self.model_signal),
                ROOT.RooArgList(self.n_signal),
            )
        else:
            self.total_model = ROOT.RooAddPdf(
                "model",
                "total model",
                ROOT.RooArgList(self.model_signal, self.model_background),
                ROOT.RooArgList(self.n_signal, self.n_background),
            )

    def fit(self):
        self._check_members()

        data_hist = ROOT.RooDataHist(
            f"{self.hist.GetName()}_roofit",
            "n-sigma distribution",
            ROOT.RooArgList(self.variable),
            ROOT.RooFit.Import(self.hist),
        )
        self.fit_results = self.total_model.fitTo(
            data_hist,
            ROOT.RooFit.Extended(),
            ROOT.RooFit.Verbose(False),
            ROOT.RooFit.PrintEvalErrors(-1),
            ROOT.RooFit.PrintLevel(-1),
            ROOT.RooFit.Range(self.variable_range[0], self.variable_range[1]),
            ROOT.RooFit.Save(),
        )
        self.frame = self.variable.frame(self.variable_range[0], self.variable_range[1])
        frame_title = f"{self.cent_label}, {self.pt_label}"
        self.frame.SetTitle(frame_title)
        histo_name = self.hist.GetName()
        frame_name = histo_name.replace("h", "fr", 1)
        self.frame.SetName(frame_name)
        data_hist.plotOn(
            self.frame, ROOT.RooFit.Name("data"), ROOT.RooFit.DrawOption("pz")
        )

        # background line
        if not self.no_bkg:
            self.total_model.plotOn(
                self.frame,
                ROOT.RooFit.Components("background"),
                ROOT.RooFit.LineStyle(ROOT.kDashed),
                ROOT.RooFit.LineColor(ROOT.kRed + 2),
                ROOT.RooFit.Name("background"),
            )

        # Total line
        self.total_model.plotOn(
            self.frame,
            ROOT.RooFit.DrawOption(""),
            ROOT.RooFit.LineColor(ROOT.kAzure + 2),
            ROOT.RooFit.Name("model"),
        )

        self.chi2_ndf = self.frame.chiSquare("model", "data")
        self.ndf = self.hist.GetNbinsX() - self.fit_results.floatParsFinal().getSize()
        self.chi2 = self.chi2_ndf * self.ndf

        # purity determination

        # total integral in the signal region
        self.tot_integral = self.hist.Integral(
            self.hist.FindBin(self.integral_range[0]),
            self.hist.FindBin(self.integral_range[1]),
        )
        self.tot_integral_error = np.sqrt(self.tot_integral)

        if self.no_bkg:
            self.bkg_integral = 0
            self.bkg_integral_error = 0
            self.purity = 1
            self.purity_error = 0

        else:
            # background integral in the signal region
            if self.exp_bkg:
                bkg_par_list = ROOT.RooArgList(self.tau_background, self.n_background)
            else:
                bkg_par_list = ROOT.RooArgList(
                    self.mu_background,
                    self.sigma_background,
                    self.alpha_left_background,
                    self.alpha_right_background,
                    self.n_background,
                )
            bkg_model = ROOT.RooAddPdf(
                "bkg_model",
                "bkg function TOF",
                ROOT.RooArgList(self.model_background),
                ROOT.RooArgList(self.n_background),
            )
            bkg_model.fixCoefNormalization(ROOT.RooArgSet(self.variable))
            func_bkg = bkg_model.asTF(ROOT.RooArgList(self.variable), bkg_par_list)
            bkg_integral_full = func_bkg.Integral(
                self.variable_range[0], self.variable_range[1]
            )
            self.bkg_integral = (
                self.n_background.getVal()
                * func_bkg.Integral(self.integral_range[0], self.integral_range[1], 0)
                / bkg_integral_full
            )
            self.bkg_integral_error = np.sqrt(self.bkg_integral)

            # purity determination
            self.purity = 1 - (self.bkg_integral / self.tot_integral)
            self.purity_error = np.hypot(
                self.bkg_integral_error / self.bkg_integral,
                self.tot_integral_error / self.tot_integral,
            )

        # create canvas
        canvas_name = frame_name.replace("fr", "c", 1) + "_Fit"
        self.canvas = ROOT.TCanvas(canvas_name, canvas_name, 800, 600)
        self.canvas.cd()

        self.info_panel = ROOT.TPaveText(0.7, 0.6, 0.9, 0.85, "brNDC")
        self.info_panel.SetBorderSize(0)
        self.info_panel.SetFillStyle(0)
        self.info_panel.SetTextAlign(12)
        self.info_panel.SetTextFont(42)
        self.info_panel.AddText(
            r"#mu_{signal} = "
            + f"{self.mu_signal.getVal():.3f} #pm {self.mu_signal.getError():.3f}"
        )
        self.info_panel.AddText(
            r"#sigma_{signal} = "
            + f"{self.sigma_signal.getVal():.3f} #pm {self.sigma_signal.getError():.3f}"
        )
        self.info_panel.AddText(
            r"#alpha_{L, sig} = "
            + f"{self.alpha_left_signal.getVal():.3f} #pm {self.alpha_left_signal.getError():.3f}"
        )
        self.info_panel.AddText(
            r"#alpha_{R, sig} = "
            + f"{self.alpha_right_signal.getVal():.3f} #pm {self.alpha_right_signal.getError():.3f}"
        )
        if not self.no_bkg:
            if self.exp_bkg:
                self.info_panel.AddText(
                    r"#tau = "
                    + f"{self.tau_background.getVal():.3f} #pm {self.tau_background.getError():.3f}"
                )
            else:
                self.info_panel.AddText(
                    r"#mu_{background} = "
                    + f"{self.mu_background.getVal():.3f} #pm {self.mu_background.getError():.3f}"
                )
                self.info_panel.AddText(
                    r"#sigma_{background} = "
                    + f"{self.sigma_background.getVal():.3f} #pm {self.sigma_background.getError():.3f}"
                )
                self.info_panel.AddText(
                    r"#alpha_{L, bkg} = "
                    + f"{self.alpha_left_background.getVal():.3f} #pm {self.alpha_left_background.getError():.3f}"
                )
                self.info_panel.AddText(
                    r"#alpha_{R, bkg} = "
                    + f"{self.alpha_right_background.getVal():.3f} #pm {self.alpha_right_background.getError():.3f}"
                )
        self.info_panel.AddText(
            r"N_{tot}"
            + f"([{self.integral_range[0]}, {self.integral_range[1]}]) =  {self.tot_integral:.1f} #pm {self.tot_integral_error:.1f}"
        )
        if not self.no_bkg:
            self.info_panel.AddText(
                r"N_{bkg}"
                + f"([{self.integral_range[0]}, {self.integral_range[1]}]) = {self.bkg_integral:.1f} #pm {self.bkg_integral_error:.1f}"
            )
        self.info_panel.AddText(
            r"purity"
            + f"([{self.integral_range[0]}, {self.integral_range[1]}]) = {self.purity:.2f}"
        )

        self.frame.Draw()
        self.info_panel.Draw()
        self.canvas.Update()
        self.canvas.SetLogy()

    def saveFrameAsPDF(self, plots_dir):
        self.canvas.SaveAs(f"{plots_dir}/{self.canvas.GetName()}.pdf")