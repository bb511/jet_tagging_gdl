#!/usr/bin/env wolframscript
(* Contact sheet of the Figure 1 candidate jets: the untransformed constituents of every
   candidate in the detector plane, marker area proportional to pT, one panel per jet.
   Usage: wolframscript -f contact_sheet.wls <fig1_candidates.csv> <out.png> *)

csv = $ScriptCommandLine[[2]];
out = $ScriptCommandLine[[3]];

data = Import[csv, "CSV"];
hdr = First[data];
rows = Rest[data];
col[name_] := First[FirstPosition[hdr, name]];

orig = Select[rows, #[[col["group"]]] === "none" &];
byCand = KeySort[GroupBy[orig, #[[col["candidate"]]] &]];

(* The jet axis from any constituent's relative angles; phi is unwrapped about it. The
   half-range of 0.5 holds every constituent of an R = 0.8 jet's core. *)
half = 0.5;
panel[rs_] := Module[{eta, phi, pt, cls, idx, ax, ay, rmax},
  eta = rs[[All, col["eta"]]];
  pt = rs[[All, col["pt"]]];
  ax = rs[[1, col["eta"]]] - rs[[1, col["etarel"]]];
  ay = Mod[rs[[1, col["phi"]]] - rs[[1, col["phirel"]]] + Pi, 2 Pi] - Pi;
  phi = ay + (Mod[# - ay + Pi, 2 Pi] - Pi) & /@ rs[[All, col["phi"]]];
  cls = rs[[1, col["class"]]];
  idx = rs[[1, col["test_index"]]];
  rmax = Max[pt];
  Graphics[{RGBColor["#0077BB"], Opacity[0.6],
    MapThread[Disk[{#1, #2}, 0.004 + 0.045 Sqrt[#3/rmax]] &, {eta, phi, pt}]},
   Frame -> True, PlotRange -> {{ax - half, ax + half}, {ay - half, ay + half}},
   AspectRatio -> 1, FrameTicks -> None,
   FrameStyle -> Directive[Black, AbsoluteThickness[0.8]],
   PlotLabel -> Style[StringJoin[cls, "  #", ToString[idx], "  n=",
      ToString[Length[rs]], "  max ", ToString[Round[rmax]], " GeV"], 13, Black],
   ImageSize -> 260]];

sheet = GraphicsGrid[Partition[panel /@ Values[byCand], 4],
   Spacings -> {20, 30}, ImageSize -> 1150];
Export[out, sheet, ImageResolution -> 144];
Print["wrote ", out, " with ", Length[byCand], " candidates"];
