(* Softmax charts of Figure 2: five classes, illustrative heights (not a model's output).
   Bars on a hairline baseline, nothing else; the class labels are an option, since the
   caption names the classes. Single charts in black (original) and the symmetry blue
   (transformed), and grouped pairs, different (the loss and the probe) and equal
   (embedding).
   Usage: wolframscript -f logits.wls <out dir> [labels] *)
outdir = $ScriptCommandLine[[2]];
labels = Length[$ScriptCommandLine] > 2;
classes = {"g", "q", "W", "Z", "t"};
pOrig = {0.04, 0.06, 0.10, 0.14, 0.66};
pTrans = {0.06, 0.14, 0.42, 0.24, 0.14};
font = "Times"; blue = RGBColor["#0077BB"];
bars[p_, colour_, x0_, wd_] := {colour,
   Table[Rectangle[{i + x0, 0}, {i + x0 + wd, p[[i]]}], {i, 5}]};
base = {Black, AbsoluteThickness[0.5], Line[{{0.5, 0}, {5.5, 0}}]};
names = If[labels,
   Table[Text[Style[classes[[i]], 7, FontFamily -> font], {i, -0.06}, {0, 1}], {i, 5}], {}];
ylo = If[labels, -0.28, -0.02];
chart[gr_] := Graphics[{gr, base, names}, PlotRange -> {{0.5, 5.5}, {ylo, 0.72}},
   PlotRangePadding -> None, AspectRatio -> 0.75, ImagePadding -> 1, ImageSize -> 54];
export[name_, g_] := (Export[FileNameJoin[{outdir, name <> ".pdf"}], g];
   Export[FileNameJoin[{outdir, name <> ".png"}], g, ImageResolution -> 200]);
export["logits_original", chart[bars[pOrig, Black, -0.25, 0.5]]];
export["logits_transformed", chart[bars[pTrans, blue, -0.25, 0.5]]];
export["logits_transformed_black", chart[bars[pTrans, Black, -0.25, 0.5]]]; (* the main rectangle: blue marks the symmetry, not the outputs *)
export["logits_pair", chart[{bars[pOrig, Black, -0.36, 0.33], bars[pTrans, blue, 0.03, 0.33]}]];
export["logits_pair_equal", chart[{bars[pOrig, Black, -0.36, 0.33], bars[pOrig, blue, 0.03, 0.33]}]];
Print["logits charts written to ", outdir];
