(* The four group panels of Figure 1: the chosen jet in the detector plane, the original
   in black and the transformed copy in colour, one panel per group, the group named
   in a tab that shares the top-left corner of the frame, a red dagger at the bottom
   right of the groups that are not a symmetry of the data, and a legend below.
   Usage: wolframscript -f panels.wls <candidates.csv> <candidate index> <out stem> *)

csv = $ScriptCommandLine[[2]];
cand = ToExpression[$ScriptCommandLine[[3]]];
stem = $ScriptCommandLine[[4]];

data = Import[csv, "CSV"]; hdr = First[data]; rows = Rest[data];
col[name_] := First[FirstPosition[hdr, name]];
rowsOf[g_] := Select[rows, #[[col["candidate"]]] == cand && #[[col["group"]]] === g &];
axis[rs_] := {rs[[1, col["eta"]]] - rs[[1, col["etarel"]]],
   Mod[rs[[1, col["phi"]]] - rs[[1, col["phirel"]]] + Pi, 2 Pi] - Pi};
unwrap[phi_, c_] := c + (Mod[phi - c + Pi, 2 Pi] - Pi);
orig = rowsOf["none"]; ptmax = Max[orig[[All, col["pt"]]]];

font = "Times";
origCol = GrayLevel[0.8]; transCol = RGBColor["#0077BB"]; daggerCol = RGBColor["#CC3311"]; (* original: a light grey shadow; transformed: the cylinder's tol blue; dagger: tol red *)
frameLine = AbsoluteThickness[0.8];
disks[rs_, c_, colour_, op_] := {colour, Opacity[op],
   MapThread[Disk[{#1, unwrap[#2, c]}, 0.004 + 0.045 Sqrt[#3/ptmax]] &,
    {rs[[All, col["eta"]]], rs[[All, col["phi"]]], rs[[All, col["pt"]]]}]};
groups = {{"so2", "SO(2)", False}, {"lorentz", "boost along z", False},
   {"so3", "SO(3)", True}, {"lorentz_iso", "isotropic boost", True}};

w = 84; gap = 12; total = 4 w + 3 gap; (* points; the cylinder is 375 pt wide *)
(* the tab's left and top edges are the frame's own lines *)
tab[text_] := Framed[Style[text, 7, FontFamily -> font, Black],
   FrameMargins -> {{2, 2}, {1, 1}}, FrameStyle -> Directive[Black, frameLine],
   Background -> White, RoundingRadius -> 0];
(* each panel is centred between the original and the transformed axis, with a half-range
   that holds both copies *)
panel[{g_, name_, dagger_}, k_] :=
  Module[{rs = rowsOf[g], a0 = axis[orig], a1, c, half, x0 = (k - 1) (w + gap)},
   a1 = axis[rs]; a1[[2]] = unwrap[a1[[2]], a0[[2]]];
   c = (a0 + a1)/2; half = Max[0.55, 0.5 Norm[a1 - a0] + 0.45];
   {Inset[Graphics[{disks[orig, c[[2]], origCol, 1], disks[rs, c[[2]], transCol, 0.8]},
      PlotRange -> {{c[[1]] - half, c[[1]] + half}, {c[[2]] - half, c[[2]] + half}},
      PlotRangePadding -> None, PlotRangeClipping -> True, AspectRatio -> 1,
      ImagePadding -> 0, Frame -> False], {x0, 0}, {Left, Bottom}, {w, w}],
    {FaceForm[None], EdgeForm[Directive[Black, frameLine]], Rectangle[{x0, 0}, {x0 + w, w}]},
    Inset[tab[name], {x0, w}, {Left, Top}],
    If[dagger, Text[Style["\[Dagger]", daggerCol, 9, FontFamily -> font], {x0 + w - 3, 3}, {1, -1}], {}],
    If[k == 1, {Text[Style["\[Eta]", 8, FontFamily -> font], {x0 + w/2, -8}],
      Text[Style["\[Phi]", 8, FontFamily -> font], {x0 - 7, w/2}]}, {}]}];

entry[text_, x_, y_] := Text[Style[text, 7, FontFamily -> font], {x, y}, {-1, 0}];
ly = -21;
legend = {{origCol, Opacity[1], Disk[{40, ly}, 2.6]}, entry["original jet", 46, ly],
   {transCol, Opacity[0.8], Disk[{120, ly}, 2.6]}, entry["transformed jet", 126, ly],
   Text[Style["\[Dagger]", daggerCol, 9, FontFamily -> font], {215, ly}, {-1, 0}],
   entry["not a symmetry of the data", 222, ly]};

fig = Graphics[{MapIndexed[panel[#1, First[#2]] &, groups], legend},
   PlotRange -> {{-14, total + 2}, {-28, w + 4}}, PlotRangePadding -> None,
   ImageSize -> {total + 16, w + 32}, ImagePadding -> 0];
Export[stem <> ".pdf", fig];
Export[stem <> ".png", fig, ImageResolution -> 250];
Print["wrote ", stem, ".{pdf,png}"];
