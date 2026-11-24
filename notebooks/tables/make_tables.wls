#!/usr/bin/env wolframscript
(* Writes the tabulars of the paper's tables, main text and appendices, from results/.
   Usage: wolframscript -f make_tables.wls <results dir> <out dir> *)

{resultsDir, outDir} = $ScriptCommandLine[[2 ;; 3]];

readTable[path_] := With[{csv = Import[FileNameJoin[{resultsDir, path}], "CSV"]},
   AssociationThread[First[csv], #] & /@ Rest[csv]];

fixed[x_, 0] := ToString[Round[x]];
fixed[x_, d_] := ToString[NumberForm[N[x], {Infinity, d}, ExponentFunction -> (Null &)]];
withUnc[mean_, std_, d_] := fixed[mean, d] <> "(" <> ToString[Round[std 10^d]] <> ")";
texLine[cells_] := StringRiffle[cells, " & "] <> " \\\\";

ns = {16, 50, 150};
findRow[tab_, model_, n_] := SelectFirst[tab, #model == model && #nconstituents == n &, Missing[]];

(* Macro accuracy in per cent and rejection at eps_S = 30 %, fold mean(std); dashes where
   the row has no run at that N. *)
perfCells[r_Missing] := {"{--}", "{--}"};
perfCells[r_] := {withUnc[100 r["test_acc_mean"], 100 r["test_acc_std"], 2],
   withUnc[r["rejection_es30_mean"], r["rejection_es30_std"], 0]};

(* bests: N -> {highest accuracy, highest rejection} of a table; matching cells go bold. *)
emph[cell_, value_, best_] := If[value === best, "\\bfseries " <> cell, cell];
perfCells[r_Missing, _] := perfCells[r];
perfCells[r_, bests_] := MapThread[emph, {perfCells[r],
    {r["test_acc_mean"], r["rejection_es30_mean"]}, Lookup[bests, r["nconstituents"], {None, None}]}];
perfLine[label_, groups_, tab_, model_, bests_ : <||>] :=
  texLine[Join[{label, groups}, Flatten[perfCells[findRow[tab, model, #], bests] & /@ ns]]];
bestAt[rows_, n_] := With[{rs = Select[rows, #nconstituents == n &]},
   {Max[Lookup[rs, "test_acc_mean"]], Max[Lookup[rs, "rejection_es30_mean"]]}];
columnBests[rows_] := AssociationMap[bestAt[rows, #] &, ns];

perfHeader = "\\toprule\n& & \\multicolumn{2}{c}{$N = 16$} & \\multicolumn{2}{c}{$N = 50$} & \
\\multicolumn{2}{c}{$N = 150$} \\\\\n\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}\n\
Model & Exact groups & {Acc.\\ (\\%)} & {$1/\\varepsilon_\\mathrm{B}$} & {Acc.\\ (\\%)} & \
{$1/\\varepsilon_\\mathrm{B}$} & {Acc.\\ (\\%)} & {$1/\\varepsilon_\\mathrm{B}$} \\\\\n\\midrule\n";
perfColumns = "@{}ll S[table-format=2.2(2)] S[table-format=3.0(2)] S[table-format=2.2(2)] \
S[table-format=4.0(3)] S[table-format=2.2(2)] S[table-format=4.0(3)]@{}";
perfTable[blocks_] := "\\begin{tabular}{" <> perfColumns <> "}\n" <> perfHeader <>
   StringRiffle[StringRiffle[#, "\n"] & /@ blocks, "\n\\midrule\n"] <>
   "\n\\bottomrule\n\\end{tabular}\n";

all4 = "SO(1,3)"; beam = "SO(2), SO(1,1)"; rot = "SO(2)";
embedded = readTable["embedded/results_table_baseline.csv"];
jetLevel = readTable["jet_level/results_table_baseline.csv"];
relative = readTable["pt_etarel_phirel/results_table_baseline.csv"];

(* label, exact groups, table, model; one list per block of Table 1 *)
embeddedSpecs = {
   {{"LorentzNet", all4, embedded, "lorentznet"},
    {"PELICAN", rot, embedded, "pelican"},
    {"PELICAN$_\\mathrm{Lorentz}$", all4, embedded, "pelican_lorentz"},
    {"L-GATr-slim", rot, embedded, "lgatr_slim"},
    {"L-GATr-slim$_\\mathrm{Lorentz}$", all4, embedded, "lgatr_slim_lorentz"},
    {"LLoCa$_\\mathrm{Lorentz}$", all4, embedded, "lloca_lorentz"},
    {"LLoCa$_z$", beam, embedded, "lloca_z"},
    {"LLoCa$_{\\mathrm{SO}(2)}$", rot, embedded, "lloca_so2"},
    {"LLoCa$_{\\mathrm{SO}(3)}$", "SO(2), SO(3)", embedded, "lloca_so3"}},
   {{"Lifted vector neurons", beam, embedded, "vn"}},
   {{"BDT, 53 observables", "{--}", jetLevel, "jet_level_bdt_all53"}}};
embeddedBests = columnBests[DeleteMissing[
    Flatten[Table[findRow[s[[3]], s[[4]], n], {s, Flatten[embeddedSpecs, 1]}, {n, ns}]]]];
embeddedTable = "\\sisetup{detect-weight}\n" <>
   perfTable[Map[perfLine[Sequence @@ #, embeddedBests] &, embeddedSpecs, {2}]];

momentsTable = perfTable[{
    {perfLine["DeepSets", rot, relative, "deepsets"],
     perfLine["DeepSets + moments", rot, embedded, "deepsets_moments"],
     perfLine["JEDI-linear", rot, relative, "jedi_linear"],
     perfLine["JEDI-linear + moments", rot, embedded, "jedi_linear_moments"]},
    {perfLine["Moments only MLP", beam, embedded, "moments_only"],
     perfLine["Lifted vector neurons", beam, embedded, "vn"]}}];

foldCell[r_] := If[r["n_folds"] == 0, "{--}",
   withUnc[100 r["test_acc_fold_mean_frac"], 100 r["test_acc_fold_std_frac"], 2]];
signed[x_, d_] := If[x >= 0, "+", ""] <> fixed[x, d];
inputLabel[r_] := If[StringStartsQ[r["inputs"], "full axis"], "full axis", "trigger"];
fullAxisQ[r_] := inputLabel[r] == "full axis";
(* The retry designs of N = 150 stay out of both tables; the text quotes the one under 100 ns. *)
fpgaRows[n_] := Select[readTable["fpga/comparison_n" <> ToString[n] <> ".csv"],
   ! StringStartsQ[#protocol, "retry"] &];
blockHead[label_, cols_] := {"\\multicolumn{" <> ToString[cols] <> "}{@{}l}{" <> label <> "} \\\\",
   "\\cmidrule{1-" <> ToString[cols] <> "}"};
fpgaTabular[columns_, header_, blocks_] := "\\begin{tabular}{" <> columns <> "}\n\\toprule\n" <>
   header <> " \\\\\n\\midrule\n" <> StringRiffle[blocks, "\n\\midrule\n"] <>
   "\n\\bottomrule\n\\end{tabular}\n";
(* The same at the full text width, the spare space spread between the columns. *)
fpgaTabularWide[columns_, header_, blocks_] := StringReplace[
   fpgaTabular["@{\\extracolsep{\\fill}}" <> StringDrop[columns, 3], header, blocks],
   {"\\begin{tabular}" -> "\\begin{tabular*}{\\linewidth}", "\\end{tabular}" -> "\\end{tabular*}"}];

(* Table 3: one row per N and input contract, JEDI-linear and the LVN side by side, both the
   most accurate design under 100 ns. JEDI-linear uses no DSP, so only the LVN's DSPs get a
   column; the groups each design holds are in the caption. The dagger marks the N = 150 LVN
   designs that alkaid placed under 100 ns and Vivado does not. *)
pairLine[n_, jedi_, vn_, first_] := texLine[{If[first, ToString[n], ""], inputLabel[vn],
    foldCell[jedi], foldCell[vn],
    fixed[jedi["latency_ns"], 0],
    fixed[vn["latency_ns"], 0] <> If[StringStartsQ[vn["protocol"], "picked"], "{$^\\dagger$}", ""],
    fixed[jedi["lut"]/1000, 0], fixed[vn["lut"]/1000, 0], ToString[vn["dsp"]]}];
pairedBlock[n_] := Module[{rows, pair},
   rows = Select[fpgaRows[n], StringStartsQ[#protocol, "best"] || StringStartsQ[#protocol, "picked"] &];
   pair[full_, first_] := With[{sel = Select[rows, fullAxisQ[#] == full &]},
     pairLine[n, SelectFirst[sel, #architecture == "JEDI-linear" &],
      SelectFirst[sel, #architecture == "VN" &], first]];
   pair[True, True] <> "\n" <> pair[False, False]];
fpgaTable = StringReplace[fpgaTabularWide[
    "@{}ll S[table-format=2.2(2)] S[table-format=2.2(2)] S[table-format=2.0] \
S[table-format=3.0] S[table-format=3.0] S[table-format=3.0] S[table-format=4.0]@{}",
    "& & \\multicolumn{2}{c}{Acc.\\ (\\%)} & \\multicolumn{2}{c}{Lat.\\ (ns)} & \
\\multicolumn{2}{c}{LUT (k)} & {DSP} \\\\\n\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \
\\cmidrule(lr){7-8} \\cmidrule(l){9-9}\n$N$ & Inputs & {JEDI} & {LVN} & \
{JEDI} & {LVN} & {JEDI} & {LVN} & {LVN}",
    pairedBlock /@ ns], "\n\\midrule\n" ~~ d : DigitCharacter.. ~~ " & " :>
    If[d == "16", "\n\\midrule\n", "\n\\addlinespace\n"] <> d <> " & "];

(* Appendix: the published JEDI-linear designs and our picks nearest their EBOPs. *)
reproLine[r_] := texLine[{inputLabel[r],
    If[StringStartsQ[r["protocol"], "published"], "published", "paper EBOPs"],
    fixed[100 r["test_acc_main_frac"], 2], foldCell[r], fixed[r["latency_ns"], 1],
    fixed[r["lut"]/1000, 1], fixed[r["ebops_main"]/1000, 1]}];
reproBlock[n_] := Module[{rows = fpgaRows[n], published},
   published = SelectFirst[rows, #source == "published" &]["nconstituents"];
   StringRiffle[Join[blockHead["$N = " <> ToString[n] <>
       If[published == n, "$", "$ (published design at $N = " <> ToString[published] <> "$)"], 7],
     reproLine /@ Select[rows, StringStartsQ[#protocol, "published"] || StringStartsQ[#protocol, "paper"] &]], "\n"]];
fpgaReproTable = fpgaTabular[
   "@{}ll S[table-format=2.2] S[table-format=2.2(2)] S[table-format=2.1] S[table-format=2.1] S[table-format=3.1]@{}",
   "Inputs & Pick & {Acc.\\ (\\%)} & {Folds (\\%)} & {Lat.\\ (ns)} & {LUT (k)} & {EBOPs (k)}",
   reproBlock /@ ns];

(* Appendix A: the clean figures of Tables 1 and 2, and the grid. *)
groups = {"so2", "lorentz", "so3"};
groupLabels = <|"so2" -> "SO(2)", "lorentz" -> "SO(1,1)", "so3" -> "SO(3)"|>;
sets = {"px_py_pz_e", "pt_eta_phi", "pt_etarel_phirel"};
setHeads = {"$[p_x, p_y, p_z, E]$", "$[p_\\mathrm{T}, \\eta, \\phi]$",
   "$[p_\\mathrm{T}, \\eta_\\mathrm{rel}, \\phi_\\mathrm{rel}]$"};
gridModels = {{"DeepSets", "deepsets"}, {"MLP", "mlp"}, {"JetFormer", "jetformer"},
   {"Linformer", "linformer"}, {"JEDI-linear", "jedi_linear"}};
readSet[prefix_, set_] := Join @@ (readTable[prefix <> set <> "/results_table_" <> # <> ".csv"] & /@
     {"baseline", "augmentation", "symm_loss"});
grid = AssociationMap[readSet["", #] &, sets];
gridRow[tab_, model_, method_, scheme_, n_] := SelectFirst[tab, #model == model && #method == method &&
     #["augmentation_scheme"] == scheme && #nconstituents == n &];
regimeRows = Join[{{"Baseline", "baseline", "none"}},
   {"Aug.\\ " <> groupLabels[#], "augmentation", #} & /@ groups,
   {"SCL " <> groupLabels[#], "symm_loss", #} & /@ groups];
tabular[columns_, header_, blocks_] := "\\begin{tabular}{" <> columns <> "}\n\\toprule\n" <> header <>
   " \\\\\n\\midrule\n" <> StringRiffle[blocks, "\n\\midrule\n"] <> "\n\\bottomrule\n\\end{tabular}\n";
(* One block of rows per model, under a line that names it; thin, so that a table of 50 rows
   stays on one page. *)
namedBlock[name_, cols_, lines_] := StringRiffle[Prepend[lines,
    "\\multicolumn{" <> ToString[cols] <> "}{@{}l}{\\emph{" <> name <> "}} \\\\"], "\n"];
ruledBlock[name_, cols_, lines_] := namedBlock[name, cols,
   Prepend[lines, "\\cmidrule{1-" <> ToString[cols] <> "}"]];

cleanCells[r_Missing] := ConstantArray["{--}", 3];
cleanCells[r_] := {withUnc[r["auc_mean"], r["auc_std"], 4],
   withUnc[100 r["tpr_eb1_mean"], 100 r["tpr_eb1_std"], 1],
   withUnc[100 r["tpr_eb10_mean"], 100 r["tpr_eb10_std"], 1]};
cleanLine[{label_, tab_, model_}] := texLine[Prepend[Flatten[cleanCells[findRow[tab, model, #]] & /@ ns], label]];
cleanSpecs = {Map[Delete[#, 2] &, embeddedSpecs[[1]]],
   {{"DeepSets", relative, "deepsets"}, {"DeepSets + moments", embedded, "deepsets_moments"},
    {"JEDI-linear", relative, "jedi_linear"}, {"JEDI-linear + moments", embedded, "jedi_linear_moments"},
    {"Moments only MLP", embedded, "moments_only"}, {"Lifted vector neurons", embedded, "vn"}},
   {{"MLP, 53 observables", jetLevel, "jet_level_all53"}, {"MLP, 52 observables", jetLevel, "jet_level_noeta52"},
    {"BDT, 53 observables", jetLevel, "jet_level_bdt_all53"}, {"BDT, 52 observables", jetLevel, "jet_level_bdt_noeta52"}}};
cleanTable = tabular[
   "@{}l *{3}{S[table-format=1.4(2)] S[table-format=2.1(2)] S[table-format=2.1(2)]}@{}",
   "& \\multicolumn{3}{c}{$N = 16$} & \\multicolumn{3}{c}{$N = 50$} & \\multicolumn{3}{c}{$N = 150$} \\\\\n\
\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}\n\
Model" <> StringRepeat[" & {AUC} & {TPR$_{1}$ (\\%)} & {TPR$_{10}$ (\\%)}", 3],
   StringRiffle[cleanLine /@ #, "\n"] & /@ cleanSpecs];

(* Three digits of the fold spread at most; a spread of ten points and more loses a decimal. *)
accCell[mean_, std_] := withUnc[100 mean, 100 std, If[100 std >= 10, 1, 2]];
gridAccLine[model_, {label_, method_, scheme_}] := texLine[Prepend[Flatten[Table[
      With[{r = gridRow[grid[s], model, method, scheme, n]}, accCell[r["test_acc_mean"], r["test_acc_std"]]],
      {s, sets}, {n, ns}]], label]];
setSpans = StringRiffle["\\multicolumn{3}{c}{" <> # <> "}" & /@ setHeads, " & "];
gridAccBlock[{name_, model_}] := ruledBlock[name, 10, gridAccLine[model, #] & /@ regimeRows];
gridAccTable = tabular["@{}l *{9}{S[table-format=2.2(3)]}@{}",
   "& " <> setSpans <> " \\\\\n\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}\n\
Regime" <> StringRepeat[" & {$N = 16$} & {50} & {150}", 3],
   gridAccBlock /@ gridModels];

(* TV in units of 1e-3 to three figures, the flip rate in per cent, the retentions as ratios. *)
tvCell[x_] := Which[x >= 100, fixed[x, 0], x >= 10, fixed[x, 1], True, fixed[x, 2]];
flipCell[x_] := If[x >= 1, fixed[x, 1], fixed[x, 2]];
probeCells[r_, g_] := {tvCell[1000 r[g <> "_tv_mean"]], flipCell[100 r[g <> "_flip_mean"]],
   fixed[r[g <> "_rejection_retention_es30_mean"], 2], fixed[r[g <> "_tpr_retention_eb1_mean"], 2],
   fixed[r[g <> "_tpr_retention_eb10_mean"], 2]};
probeColumns = "@{}ll *{15}{r}@{}";
probeHeader[first_, second_] := "& & \\multicolumn{5}{c}{$N = 16$} & \\multicolumn{5}{c}{$N = 50$} & \
\\multicolumn{5}{c}{$N = 150$} \\\\\n\\cmidrule(lr){3-7} \\cmidrule(lr){8-12} \\cmidrule(lr){13-17}\n" <>
   first <> " & " <> second <> StringRepeat[
    " & {TV} & {Flip} & {$\\rho_\\mathrm{B}$} & {$\\rho_{1}$} & {$\\rho_{10}$}", 3];
probeLine[tab_, model_, g_, first_, {label_, method_, scheme_}] := texLine[Join[
    {If[first, groupLabels[g], ""], label},
    Flatten[probeCells[gridRow[tab, model, method, scheme, #], g] & /@ ns]]];
probeTriplet[tab_, model_, g_] := MapThread[probeLine[tab, model, g, #1, #2] &, {{True, False, False},
    {{"Base", "baseline", "none"}, {"Aug.", "augmentation", g}, {"SCL", "symm_loss", g}}}];
gridProbeBlock[set_][{name_, model_}] := ruledBlock[name, 17,
   Flatten[probeTriplet[grid[set], model, #] & /@ groups]];
gridProbeTable[set_] := tabular[probeColumns, probeHeader["Group", "Regime"],
   gridProbeBlock[set] /@ gridModels];

embeddedProbeLine[name_, model_, g_] := texLine[Join[{If[g == First[groups], name, ""], groupLabels[g]},
    Flatten[probeCells[findRow[embedded, model, #], g] & /@ ns]]];
embeddedProbeRow[spec_] := embeddedProbeLine[First[spec], Last[spec], #] & /@ groups;
embeddedProbeBlock[specs_] := StringRiffle[Flatten[embeddedProbeRow /@ specs], "\n"];
embeddedProbeTable = tabular[probeColumns, probeHeader["Model", "Group"], embeddedProbeBlock /@ {
     embeddedSpecs[[1]],
     {{"Lifted vector neurons", "vn"}, {"DeepSets + moments", "deepsets_moments"},
      {"JEDI-linear + moments", "jedi_linear_moments"}, {"Moments only MLP", "moments_only"}}}];

(* Appendix B: the replica's figures are those the papers quote, per run. *)
lit = readTable["top_tagging/literature_figures.csv"];
published = With[{csv = Import[FileNameJoin[{resultsDir, "top_tagging/published.csv"}], "CSV", "Numeric" -> False]},
   AssociationThread[First[csv], #] & /@ Rest[csv]];
pubRow[tagger_] := SelectFirst[published, #tagger == tagger &];
(* The published digits as printed: a fraction 0.942 is 94.2 %, 0.9420 is 94.20 %. *)
pubAcc[s_] := If[s == "", "{--}", StringTake[s, {3, 4}] <> "." <> StringDrop[s, 4]];
pubCells[r_] := {pubAcc[r["acc"]], If[r["auc"] == "", "{--}", r["auc"]],
   r["rejection_es30"] <> If[r["rejection_es30_err"] == "", "", "(" <> r["rejection_es30_err"] <> ")"]};
foldStat[rs_, key_] := {Mean[Lookup[rs, key]], StandardDeviation[Lookup[rs, key]]};
litFolds[table_, row_, regime_] := Select[lit, #table == table && #row == row && #regime == regime && #run != "main" &];
litCells[rs_] := {accCell @@ foldStat[rs, "acc"], withUnc[Sequence @@ foldStat[rs, "auc"], 4],
   withUnc[Sequence @@ foldStat[rs, "rejection_es30"], 0]};
calibLine[{label_, row_, tagger_}] := texLine[Join[{label}, litCells[litFolds["embedded", row, "baseline"]],
    If[tagger === None, ConstantArray["{--}", 3], pubCells[pubRow[tagger]]]]];
expertLine[{label_, tagger_}] := texLine[Join[{label}, ConstantArray["{--}", 3], pubCells[pubRow[tagger]]]];
topCalibTable = tabular[
   "@{}l S[table-format=2.2(2)] S[table-format=1.4(2)] S[table-format=4.0(3)] \
S[table-format=2.2] S[table-format=1.4] S[table-format=4.0(3)]@{}",
   "& \\multicolumn{3}{c}{Ours} & \\multicolumn{3}{c}{Published} \\\\\n\
\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\nModel" <> StringRepeat[
     " & {Acc.\\ (\\%)} & {AUC} & {$1/\\varepsilon_\\mathrm{B}$}", 2],
   {StringRiffle[calibLine /@ {
       {"LorentzNet", "lorentznet", "LorentzNet"}, {"PELICAN", "pelican", "PELICAN 48k"},
       {"PELICAN$_\\mathrm{Lorentz}$", "pelican_lorentz", None},
       {"L-GATr-slim", "lgatr_slim", "L-GATr-slim"},
       {"L-GATr-slim$_\\mathrm{Lorentz}$", "lgatr_slim_lorentz", None},
       {"LLoCa$_\\mathrm{Lorentz}$", "lloca_lorentz", "LLoCa-Transformer"},
       {"LLoCa$_z$", "lloca_z", None}, {"LLoCa$_{\\mathrm{SO}(2)}$", "lloca_so2", None},
       {"LLoCa$_{\\mathrm{SO}(3)}$", "lloca_so3", None}}, "\n"],
    StringRiffle[calibLine /@ {
       {"DeepSets + moments", "deepsets_moments", None}, {"JEDI-linear + moments", "jedi_linear_moments", None},
       {"Moments only MLP", "moments_only", None}, {"Lifted vector neurons", "vn", None}}, "\n"],
    StringRiffle[expertLine /@ {
       {"$N$-subjettiness DNN", "N-subjettiness 8 (DNN)"}, {"Linear EFPs", "EFP (linear)"},
       {"BIP (MLP)", "BIP (MLP)"}, {"DisCo-FFS", "DisCo-FFS"}}, "\n"]}];

topGrid = AssociationMap[readSet["top_tagging/", #] &, sets];
regimeTag[method_, scheme_] := Switch[method, "baseline", "baseline", "augmentation", "aug_" <> scheme,
    "symm_loss", "symm_" <> scheme];
topCells[set_, model_, method_, scheme_, g_] := With[{rs = litFolds[set, model, regimeTag[method, scheme]],
    r = gridRow[topGrid[set], model, method, scheme, 100]}, {accCell @@ foldStat[rs, "acc"],
    withUnc[Sequence @@ foldStat[rs, "rejection_es30"], 0], tvCell[1000 r[g <> "_tv_mean"]]}];
topLine[model_, g_, first_, {label_, method_, scheme_}] := texLine[Join[{If[first, groupLabels[g], ""], label},
    Flatten[topCells[#, model, method, scheme, g] & /@ sets]]];
topTriplet[model_, g_] := MapThread[topLine[model, g, #1, #2] &, {{True, False, False},
    {{"Base", "baseline", "none"}, {"Aug.", "augmentation", g}, {"SCL", "symm_loss", g}}}];
topGridBlock[{name_, model_}] := ruledBlock[name, 11, Flatten[topTriplet[model, #] & /@ groups]];
topGridTable = tabular[
   "@{}ll *{3}{S[table-format=2.2(3)] S[table-format=4.0(3)] r}@{}",
   "& & " <> setSpans <> " \\\\\n\
\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11}\nGroup & Regime" <> StringRepeat[
     " & {Acc.\\ (\\%)} & {$1/\\varepsilon_\\mathrm{B}$} & {TV}", 3],
   topGridBlock /@ gridModels];

topFpga = readTable["fpga/comparison_top_tagging_n100.csv"];
topFpgaLine[stem_, inputs_, model_] := With[{r = SelectFirst[topFpga, StringStartsQ[#["design_id"], stem] &]},
   texLine[{inputs, model, fixed[100 r["test_acc_main_frac"], 2], foldCell[r], fixed[r["auc_fold_mean"], 4],
     withUnc[r["rejection_es30_fold_mean"], r["rejection_es30_fold_std"], 0], ToString[r["latency_cycles"]],
     fixed[r["latency_ns"], 1], fixed[r["lut"]/1000, 0], ToString[r["dsp"]], fixed[r["ebops_main"]/1000, 0]}]];
topFpgaTable = tabular[
   "@{}ll S[table-format=2.2] S[table-format=2.2(1)] S[table-format=1.4] S[table-format=3.0(2)] \
S[table-format=2.0] S[table-format=3.1] S[table-format=3.0] S[table-format=4.0] S[table-format=3.0]@{}",
   "Inputs & Model & {Acc.\\ (\\%)} & {Folds (\\%)} & {AUC} & {$1/\\varepsilon_\\mathrm{B}$} & {Cycles} & \
{Lat.\\ (ns)} & {LUT (k)} & {DSP} & {EBOPs (k)}",
   {StringRiffle[{topFpgaLine["jedi_keptaxis_triggerwords", "trigger", "JEDI-linear"],
      topFpgaLine["vn_noaxis_triggerwords", "", "LVN"]}, "\n"],
    StringRiffle[{topFpgaLine["jedi_keptaxis_trainedwords", "trained", "JEDI-linear"],
      topFpgaLine["vn_keptaxis_trainedwords", "", "LVN"]}, "\n"]}];

(* Appendix C: size and cost of every row, the grid rows on the relative coordinates. *)
params = readTable["params.csv"];
paramsOf[model_, label_, n_] := SelectFirst[params, #model == model && #label == label &&
     #variant == "baseline" && #nconstituents == n &, Missing[]];
sci[x_] := With[{e = Floor[Log10[Round[x, 10^(Floor[Log10[x]] - 1)]]]},
   "$" <> fixed[x/10^e, 1] <> "\\times 10^{" <> ToString[e] <> "}$"];
costCells[_Missing, _] := ConstantArray["{--}", 3];
costCells[r_, p_] := {fixed[p["params"]/1000, 1], sci[r["flops_mean"]], sci[r["ebops_mean"]]};
costLine[{label_, inputs_, tab_, model_, paramLabel_}] := texLine[Join[{label, inputs},
    Flatten[costCells[findRow[tab, model, #], paramsOf[model, paramLabel, #]] & /@ ns]]];
withCosts[inputs_, tab_, paramLabel_][{label_, model_}] := {label, inputs, tab, model, paramLabel};
costTable = tabular["@{}ll *{9}{r}@{}",
   "& & \\multicolumn{3}{c}{$N = 16$} & \\multicolumn{3}{c}{$N = 50$} & \\multicolumn{3}{c}{$N = 150$} \\\\\n\
\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11}\nModel & Inputs" <> StringRepeat[
     " & {Par.\\ (k)} & {FLOPs} & {EBOPs}", 3],
   {StringRiffle[costLine /@ (withCosts["relative", relative, "pt_etarel_phirel"] /@ gridModels), "\n"],
    StringRiffle[costLine /@ (withCosts["Cartesian", embedded, "default"] /@ Map[#[[{1, 4}]] &, embeddedSpecs[[1]]]), "\n"],
    StringRiffle[costLine /@ Join[
       {{"Lifted vector neurons", "cylindrical", embedded, "vn", "default"}},
       withCosts["relative", embedded, "default"] /@ {{"DeepSets + moments", "deepsets_moments"},
         {"JEDI-linear + moments", "jedi_linear_moments"}, {"Moments only MLP", "moments_only"}}], "\n"],
    StringRiffle[costLine /@ {{"MLP, 53 observables", "jet level", jetLevel, "jet_level_all53", "default"},
       {"MLP, 52 observables", "jet level", jetLevel, "jet_level_noeta52", "default"}}, "\n"]}];

(* Appendix D: what Table 3 leaves out of its twelve designs, and the latency ablation. *)
designLine[r_, n_, inputs_, model_] := texLine[{n, inputs, model, fixed[100 r["test_acc_main_frac"], 2],
    ToString[r["latency_cycles"]], fixed[r["wns_ns"], 2], fixed[r["ff"]/1000, 0], fixed[r["ebops_main"]/1000, 0]}];
designRows[n_] := Select[fpgaRows[n], StringStartsQ[#protocol, "best"] || StringStartsQ[#protocol, "picked"] &];
designBlock[n_] := With[{rs = designRows[n]}, StringRiffle[Flatten[Table[
      With[{sel = Select[rs, fullAxisQ[#] == full &]}, {
        designLine[SelectFirst[sel, #architecture == "JEDI-linear" &],
         If[full, ToString[n], ""], If[full, "full axis", "trigger"], "JEDI-linear"],
        designLine[SelectFirst[sel, #architecture == "VN" &], "", "", "LVN"]}], {full, {True, False}}]], "\n"]];
retryRow = SelectFirst[readTable["fpga/comparison_n150.csv"], StringEndsQ[#["design_id"], "retry_e894"] &];
fpgaDesignsTable = tabular[
   "@{}lll S[table-format=2.2] S[table-format=2.0] S[table-format=-1.2] S[table-format=3.0] S[table-format=3.0]@{}",
   "$N$ & Inputs & Model & {Acc.\\ (\\%)} & {Cycles} & {WNS (ns)} & {FF (k)} & {EBOPs (k)}",
   Join[designBlock /@ ns, {designLine[retryRow, "150", "full axis", "LVN$^\\ddagger$"]}]];

history = readTable["fpga/history/vivado_history.csv"];
quantised = readTable["quantised/results_table_baseline.csv"];
ablationLine[{label_, campaign_, design_}] := With[{
    h = SelectFirst[history, #campaign == campaign && #design == design &]},
   With[{q = SelectFirst[quantised, #["ebops_mean"] == h["ebops"] && #nconstituents == 16 &]},
    texLine[{label, fixed[100 h["acc_micro_alkaid"], 2], fixed[q["rejection_es30_mean"], 0],
      ToString[h["latency_cycles"]], fixed[h["latency_ns"], 0], fixed[h["lut"]/1000, 0], ToString[h["dsp"]],
      fixed[h["ebops"]/1000, 0]}]]];
fpgaAblationTable = tabular[
   "@{}l S[table-format=2.2] S[table-format=3.0] S[table-format=2.0] S[table-format=3.0] S[table-format=2.0] \
S[table-format=3.0] S[table-format=3.0]@{}",
   "Network & {Acc.\\ (\\%)} & {$1/\\varepsilon_\\mathrm{B}$} & {Cycles} & {Lat.\\ (ns)} & {LUT (k)} & {DSP} & {EBOPs (k)}",
   {StringRiffle[ablationLine /@ {
       {"One block", "hgq_scan_lc42", "beta3e-07"},
       {"without the gate", "hgq_ablation", "nogate_dc0"},
       {"and the 64-wide head layer", "hgq_ablation", "nogate_h32_dc0"},
       {"and the second MLP layer", "hgq_ablation", "nogate_h32_mlp1_dc0"},
       {"and the stream", "hgq_ablation", "nogate_h32_mlp1_nostream_dc0"},
       {"and the tensor invariants", "hgq_ablation", "nogate_h32_mlp1_nostream_notensor_dc0"}}, "\n"]}];

Export[FileNameJoin[{outDir, #1}], #2, "Text"] & @@@ Join[{{"embedded.tex", embeddedTable},
   {"moments.tex", momentsTable}, {"fpga.tex", fpgaTable}, {"fpga_repro.tex", fpgaReproTable},
   {"clean.tex", cleanTable}, {"grid_accuracy.tex", gridAccTable}, {"embedded_probes.tex", embeddedProbeTable},
   {"top_calibration.tex", topCalibTable}, {"top_grid.tex", topGridTable}, {"top_fpga.tex", topFpgaTable},
   {"costs.tex", costTable}, {"fpga_designs.tex", fpgaDesignsTable}, {"fpga_ablation.tex", fpgaAblationTable}},
  {"grid_probes_" <> #  <> ".tex", gridProbeTable[#]} & /@ sets];
