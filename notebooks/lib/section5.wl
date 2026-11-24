(* ::Package:: *)

(* section5.wl -- data and figures of section 5 ("What invariance buys"): Figure 3, the
   regimes against the probe, and Figure 4, accuracy against EBOPs. Loaded after
   style.wl; reads the results tables under resultsDir. Accuracies are the macro
   accuracies of the results tables, fold means, in per cent. *)

readTable[path_] := With[{csv = Import[FileNameJoin[{resultsDir, path}], "CSV"]},
   AssociationThread[First[csv], #] & /@ Rest[csv]];
findRow[tab_, model_, n_] := SelectFirst[tab, #model == model && #nconstituents == n &, Missing[]];

featureSets = {"px_py_pz_e", "pt_eta_phi", "pt_etarel_phirel"};
setLabel = <|"px_py_pz_e" -> "Cartesian", "pt_eta_phi" -> "cylindrical",
   "pt_etarel_phirel" -> "cylindrical\:2010relative"|>;
ns = {16, 50, 150};

(* One point per learned regime: the TV under the trained group relative to the
   baseline's, and the accuracy moved in per cent. SO(2) on the relative coordinates is an
   exact no-op (TV zero in both), so it is left out. *)
regimePoint[base_, r_] := With[{b = findRow[base, r["model"], r["nconstituents"]],
    g = r["augmentation_scheme"]},
   <|"method" -> r["method"], "group" -> g, "model" -> r["model"],
    "n" -> r["nconstituents"], "ratio" -> r[g <> "_tv_mean"]/b[g <> "_tv_mean"],
    "dacc" -> 100 (r["test_acc_mean"] - b["test_acc_mean"])|>];
regimePoints[set_] := With[{base = readTable[set <> "/results_table_baseline.csv"]},
   Select[regimePoint[base, #] & /@ Join[readTable[set <> "/results_table_augmentation.csv"],
       readTable[set <> "/results_table_symm_loss.csv"]],
    ! (set == "pt_etarel_phirel" && #group == "so2") &]];

(* The rows of Figure 4 at one N: label, family (marker), exact groups (colour), table. *)
costSpecs := Join[
   {#, "grid", "so2", "pt_etarel_phirel/results_table_baseline.csv"} & /@
    {"deepsets", "mlp", "jetformer", "linformer", "jedi_linear"},
   {{"deepsets_moments", "moments", "so2"}, {"jedi_linear_moments", "moments", "so2"},
     {"moments_only", "moments", "lorentz"}, {"vn", "vn", "lorentz"},
     {"pelican", "embedded", "so2"}, {"lgatr_slim", "embedded", "so2"},
     {"lloca_so2", "embedded", "so2"}, {"lloca_z", "embedded", "lorentz"},
     {"lloca_so3", "embedded", "so3"}, {"lorentznet", "embedded", "all"},
     {"pelican_lorentz", "embedded", "all"}, {"lgatr_slim_lorentz", "embedded", "all"},
     {"lloca_lorentz", "embedded", "all"}} /. {m_, f_, g_} :>
    {m, f, g, "embedded/results_table_baseline.csv"},
   {{"jet_level_all53", "jetlevel", "none", "jet_level/results_table_baseline.csv"}}];
costPoint[{m_, f_, g_, path_}, n_] := With[{r = findRow[readTable[path], m, n]},
   If[MissingQ[r], Nothing, <|"model" -> m, "family" -> f, "group" -> g,
     "ebops" -> r["ebops_mean"], "acc" -> 100 r["test_acc_mean"],
     "std" -> 100 r["test_acc_std"]|>]];
costPoints[n_] := costPoint[#, n] & /@ costSpecs;

(* Markers in points; open markers are white inside. *)
markerShape = <|"circle" -> Disk[{0, 0}, 1],
   "square" -> Rectangle[{-0.85, -0.85}, {0.85, 0.85}],
   "diamond" -> Polygon[{{0, 1.25}, {1.05, 0}, {0, -1.25}, {-1.05, 0}}],
   "triangle" -> Polygon[{{0, 1.2}, {1.1, -0.75}, {-1.1, -0.75}}],
   "star" -> Polygon[Table[(If[EvenQ[k], 1.45, 0.6]) {Sin[k Pi/5], Cos[k Pi/5]}, {k, 0, 9}]]|>;
marker[shape_, colour_, filled_, size_ : 5.5, opacity_ : 1, edge_ : 0.7] := Graphics[{
    FaceForm[Directive[Opacity[opacity], If[filled, colour, White]]],
    EdgeForm[Directive[Opacity[opacity], If[filled, Darker[colour, 0.25], colour],
      AbsoluteThickness[edge]]],
    markerShape[shape]}, ImageSize -> size, PlotRangePadding -> Scaled[0.12]];

(* Log ticks on a Log10 axis: every decade ticked, every step-th labelled, short
   unlabelled minor ticks. *)
logTicks[{lo_, hi_}, labelled_, step_ : 1] := Join[
   Table[{k, If[labelled && Mod[k, step] == 0, text[Superscript[10, minus[k]], 8], ""], {0.025, 0}},
    {k, Ceiling[lo], Floor[hi]}],
   Select[Flatten[Table[{Log10[m 10^k], "", {0.012, 0}}, {k, Floor[lo], Ceiling[hi]}, {m, 2, 9}], 1],
    lo <= #[[1]] <= hi &]];
(* typographic minus, which the operator substitution switched off in style.wl leaves alone *)
minus[x_] := StringReplace[ToString[x], "-" -> "\:2212"];
linTicks[{lo_, hi_}, step_, labelled_] :=
  Table[{y, If[labelled, text[minus[Round[y]], 8], ""], {0.025, 0}}, {y, step Ceiling[lo/step], hi, step}];

(* The feature set or N in a tab that shares the frame's top-left corner, as in Figure 1. *)
tab[label_] := Framed[text[label, 7], FrameMargins -> {{2, 2}, {1, 1}},
   FrameStyle -> frameStyle, Background -> White, RoundingRadius -> 0];

(* A framed panel of size {w, h} in points with fixed paddings, so that panels inset side
   by side keep their frames aligned. *)
panel[layers_, {xr_, yr_}, {xt_, yt_}, label_, {w_, h_}, pad_] := Graphics[layers,
   PlotRange -> {xr, yr}, PlotRangePadding -> None, PlotRangeClipping -> True,
   AspectRatio -> Full, Frame -> True, FrameStyle -> frameStyle, FrameTicks -> {{yt, None}, {xt, None}},
   FrameTicksStyle -> Directive[Black, 8], BaseStyle -> baseStyle,
   Epilog -> Inset[tab[label], Scaled[{0, 1}], {Left, Top}],
   ImageSize -> {w, h} + Total /@ pad, ImagePadding -> pad];

legendEntry[g_, label_, x_, y_] := {Inset[g, {x, y + 0.7}], Text[text[label, 7], {x + 5, y}, {-1, 0}]};
legendRow[entries_, xs_, y_] := MapThread[legendEntry[#1[[1]], #1[[2]], #2, y] &, {entries, xs}];

(* Legend rows, each centred on x on its own: an entry is as wide as its marker, 5 pt and
   its label, with 12 pt between entries. *)
labelWidth[label_] := First[Rasterize[text[label, 7], "RasterSize", ImageResolution -> 72]];
legendCentred[rows_, x_, ys_] := MapThread[With[{widths = labelWidth[#[[2]]] + 8 & /@ #1},
     legendRow[#1, x - (Total[widths] + 12 (Length[widths] - 1))/2 + 3 +
       Most[FoldList[Plus, 0, widths + 12]], #2]] &, {rows, ys}];

(* Three panels across the text width. The k-th frame starts at left + (k-1)(w+gap) and
   its panel carries lefts[[k]] of padding for its tick labels; the legend rows sit
   below the shared x label. *)
row3[panels_, {w_, h_}, {left_, gap_}, lefts_, {xlabel_, ylabel_}, legend_, legendHeight_] :=
  With[{y0 = legendHeight + 30}, Graphics[{
     MapIndexed[Inset[#1, {left + (#2[[1]] - 1) (w + gap) - lefts[[#2[[1]]]], y0 - 16},
        {Left, Bottom}] &, panels],
     Text[text[xlabel], {left + 1.5 w + gap, legendHeight + 8}],
     Text[text[ylabel], {5, y0 + h/2}, Center, {0, 1}], legend},
    PlotRange -> {{0, fullWidth}, {-4, y0 + h + 4}}, PlotRangePadding -> None,
    ImageSize -> {fullWidth, y0 + h + 8}, ImagePadding -> 0]];
row3Panels[make_, items_, size_, lefts_] :=
  MapIndexed[make[#1, #2[[1]], size, {{lefts[[#2[[1]]]], 2}, {16, 2}}] &, items];

(* Figure 3: Delta accuracy against log10 of the TV ratio, one panel per feature set,
   colour the trained group, filled circles augmentation, open triangles SCL, the SO(3)
   control faded and drawn first so the symmetries stay on top. *)
fig3XRange = {-3.6, 0.5}; fig3YRange = {-33, 8};
fig3Marker[p_] := With[{aug = p["method"] == "augmentation"},
   marker[If[aug, "circle", "triangle"], groupColour[p["group"]], aug, 4.5,
    If[p["group"] == "so3", 0.35, 1], 0.5]];
fig3Layer[p_] := Inset[fig3Marker[p], {Log10[Max[p["ratio"], 10^-3.5]], p["dacc"]}];
fig3Points[set_] := SortBy[regimePoints[set], #group != "so3" &];
zeroGuides = {guideStyle, InfiniteLine[{{0, 0}, {1, 0}}], InfiniteLine[{{0, 0}, {0, 1}}]};
fig3Panel[set_, k_, size_, pad_] := panel[
   {zeroGuides, fig3Layer /@ fig3Points[set]},
   {fig3XRange, fig3YRange}, {logTicks[fig3XRange, True], linTicks[fig3YRange, 10, k == 1]},
   setLabel[set], size, pad];

fig3Legend = legendRow[Join[
    {marker["circle", groupColour[#], True, 4.5, If[# == "so3", 0.35, 1], 0.5], groupLabel[#]} & /@
     {"so2", "lorentz", "so3"}, {{marker["circle", Black, True, 4.5, 1, 0.5], "augmentation"},
     {marker["triangle", Black, False, 4.5, 1, 0.5], "SCL"}}], {60, 105, 155, 215, 285}, 4];
fig3[] := With[{w = 116, h = 100, left = 28, gap = 8, lefts = {28, 2, 2}},
   row3[row3Panels[fig3Panel, featureSets, {w, h}, lefts], {w, h}, {left, gap}, lefts,
    {"TV under the trained group, relative to the baseline", "\[CapitalDelta] accuracy (%)"},
    fig3Legend, 8]];

(* Figure 4: accuracy against log10 EBOPs, one panel per N with its own accuracy range;
   colour the groups a row holds exactly, the marker its family; as in Figure 3, the
   markers are small, and the rows holding SO(3) or SO(1,3), neither a symmetry of the
   data, are faded and drawn first. *)
familyShape = <|"grid" -> "circle", "moments" -> "square", "vn" -> "star",
   "embedded" -> "diamond", "jetlevel" -> "triangle"|>;
familyLabel = <|"grid" -> "FPGA architectures", "moments" -> "moments",
   "vn" -> "vector neurons", "embedded" -> "symmetry\:2010embedded", "jetlevel" -> "MLP, 53 observables"|>;
fig4Colour[g_] := Lookup[groupColour, g, tol["grey"]];
fig4XRange = {5.3, 11.6};
fig4YRange[pts_] := With[{a = Lookup[pts, "acc"]}, {Floor[Min[a] - 1], Ceiling[Max[a] + 1]}];
fig4Opacity[g_] := If[MemberQ[{"so3", "all"}, g], 0.35, 1];
fig4Layer[p_] := Inset[marker[familyShape[p["family"]], fig4Colour[p["group"]], True,
    If[p["family"] == "vn", 6, 4.5], fig4Opacity[p["group"]], 0.5], {Log10[p["ebops"]], p["acc"]}];
fig4Panel[n_, k_, size_, pad_] := With[{pts = SortBy[costPoints[n], {#family == "vn" &, fig4Opacity[#group] == 1 &}],
    yr = fig4YRange[costPoints[n]]},
   panel[fig4Layer /@ pts, {fig4XRange, yr}, {logTicks[fig4XRange, True, 2], linTicks[yr, 2, True]},
    Row[{Style["N", Italic], " = ", n}], size, pad]];
fig4Legend[x_] := legendCentred[{
    {marker["circle", fig4Colour[#], True, 4.5, fig4Opacity[#], 0.5], groupLabel[#]} & /@ {"so2", "lorentz", "so3", "all"},
    {marker[familyShape[#], Black, False, If[# == "vn", 6, 4.5], 1, 0.5], familyLabel[#]} & /@
     {"grid", "moments", "vn", "embedded", "jetlevel"}}, x, {14, 4}];
fig4[] := With[{w = 107, h = 100, left = 28, gap = 22, lefts = {14, 18, 18}},
   row3[row3Panels[fig4Panel, ns, {w, h}, lefts], {w, h}, {left, gap}, lefts,
    {"EBOPs per jet", "Accuracy (%)"}, fig4Legend[left + 1.5 w + gap], 18]];

(* Readings: the numbers the text of section 5 quotes. *)

(* Per feature set, method and trained group: the range and median of the accuracy moved
   (points) and of the TV ratio. *)
regimeSummary[] := Dataset[Flatten[Table[With[{pts = Select[regimePoints[s], #method == m && #group == g &]},
      If[pts === {}, Nothing, <|"set" -> s, "method" -> m, "group" -> g,
        "dacc" -> Round[MinMax[Lookup[pts, "dacc"]], 0.1], "dacc median" -> Round[Median[Lookup[pts, "dacc"]], 0.01],
        "ratio" -> Round[MinMax[Lookup[pts, "ratio"]], 0.001], "ratio median" -> Round[Median[Lookup[pts, "ratio"]], 0.01]|>]],
     {s, featureSets}, {m, {"augmentation", "symm_loss"}}, {g, {"so2", "lorentz", "so3"}}], 2]];

(* Per architecture and N: the baseline accuracy on each feature set. *)
coordinateSummary[] := With[{base = AssociationMap[readTable[# <> "/results_table_baseline.csv"] &, featureSets]},
   Dataset[Flatten[Table[<|"model" -> m, "N" -> n,
       Sequence @@ Table[s -> Round[100 findRow[base[s], m, n]["test_acc_mean"], 0.01], {s, featureSets}]|>,
      {m, {"deepsets", "mlp", "jetformer", "linformer", "jedi_linear"}}, {n, ns}]]]];

(* The twins: accuracy of a row minus that of its twin with a larger exact group, in points. *)
twins = {{"lloca_z", "lloca_so2"}, {"pelican", "pelican_lorentz"},
   {"lgatr_slim", "lgatr_slim_lorentz"}, {"lloca_so2", "lloca_so3"}, {"lloca_z", "lloca_lorentz"}};
twinDeltas[] := With[{emb = readTable["embedded/results_table_baseline.csv"]},
   Dataset[Table[<|"pair" -> StringRiffle[t, " - "], Sequence @@ Table[ToString[n] ->
         Round[100 (findRow[emb, t[[1]], n]["test_acc_mean"] - findRow[emb, t[[2]], n]["test_acc_mean"]), 0.01],
        {n, ns}]|>, {t, twins}]]];

(* The best embedded rows at N = 150 against the ceiling row of Table 1, the BDT on all 53
   observables: accuracy in points, AUC, and the rejection ratio. *)
ceilingComparison[] := With[{emb = readTable["embedded/results_table_baseline.csv"],
    bdt = findRow[readTable["jet_level/results_table_baseline.csv"], "jet_level_bdt_all53", 150]},
   Dataset[Table[With[{r = findRow[emb, m, 150]}, <|"model" -> m,
       "dacc" -> Round[100 (r["test_acc_mean"] - bdt["test_acc_mean"]), 0.01],
       "dauc" -> Round[r["auc_mean"] - bdt["auc_mean"], 0.0001],
       "rejection ratio" -> Round[r["rejection_es30_mean"]/bdt["rejection_es30_mean"], 0.01]|>],
     {m, {"pelican", "lgatr_slim", "lloca_z", "vn"}}]]];
