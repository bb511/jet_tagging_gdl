(* Corner panels of Figure 2: the chosen jet in black and its
   copy under the boost along z in the symmetry blue, both framed identically so that
   the shift is visible. The figure is illustrative, so one group serves.
   Usage: wolframscript -f jet_panels.wls <candidates.csv> <candidate index> <out dir> *)
csv = $ScriptCommandLine[[2]];
cand = ToExpression[$ScriptCommandLine[[3]]];
outdir = $ScriptCommandLine[[4]];
group = "lorentz";
data = Import[csv, "CSV"]; hdr = First[data]; rows = Rest[data];
col[name_] := First[FirstPosition[hdr, name]];
rowsOf[g_] := Select[rows, #[[col["candidate"]]] == cand && #[[col["group"]]] === g &];
axis[rs_] := {rs[[1, col["eta"]]] - rs[[1, col["etarel"]]],
   Mod[rs[[1, col["phi"]]] - rs[[1, col["phirel"]]] + Pi, 2 Pi] - Pi};
unwrap[phi_, c_] := c + (Mod[phi - c + Pi, 2 Pi] - Pi);
orig = rowsOf["none"]; ptmax = Max[orig[[All, col["pt"]]]];
origCol = Black; blue = RGBColor["#0077BB"]; (* the original black, the symmetry Figure 1's blue *)
disks[rs_, c_, colour_, op_] := {colour, Opacity[op],
   MapThread[Disk[{#1, unwrap[#2, c]}, 0.006 + 0.07 Sqrt[#3/ptmax]] &,
    {rs[[All, col["eta"]]], rs[[All, col["phi"]]], rs[[All, col["pt"]]]}]};
frame[gr_, range_] := Graphics[gr, PlotRange -> range, PlotRangePadding -> None,
   PlotRangeClipping -> True, AspectRatio -> 1, Frame -> True, FrameTicks -> None,
   FrameStyle -> Directive[Black, AbsoluteThickness[0.8]], ImagePadding -> 1, ImageSize -> 72];
export[name_, g_] := (Export[FileNameJoin[{outdir, name <> ".pdf"}], g];
   Export[FileNameJoin[{outdir, name <> ".png"}], g, ImageResolution -> 200]);
a0 = axis[orig]; rs = rowsOf[group]; a1 = axis[rs]; a1[[2]] = unwrap[a1[[2]], a0[[2]]];
c = (a0 + a1)/2; half = Max[0.55, 0.5 Norm[a1 - a0] + 0.45];
range = {{c[[1]] - half, c[[1]] + half}, {c[[2]] - half, c[[2]] + half}};
export["jet_original", frame[disks[orig, c[[2]], origCol, 0.85], range]];
export["jet_" <> group, frame[disks[rs, c[[2]], blue, 0.8], range]];
(* a second draw for the augmentation thumbnail: a boost along z of rapidity xi2 shifts every
   massless constituent by -xi2 in eta and leaves pT alone, so it needs no export; xi2 = 0.4
   keeps the copy inside the shared frame, between the original and the first copy *)
xi2 = 0.4; rs2 = orig; rs2[[All, col["eta"]]] -= xi2;
export["jet_" <> group <> "_2", frame[disks[rs2, c[[2]], blue, 0.8], range]];
Print["jet panels written to ", outdir];
