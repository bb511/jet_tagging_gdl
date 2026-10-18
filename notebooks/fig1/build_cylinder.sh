#!/bin/bash
# build_cylinder.sh <candidates.csv> <candidate> <stem> [nsub]: the cylinder of
# cylinder_template.tex with the chosen jet projected on the barrel (a cone from the IP
# to its footprint, the constituents as discs in the tangent plane) and one smaller cone per
# subjet: the constituents are declustered with exclusive kT (R = 0.8) down to nsub
# (default 3) subjets, and each gets a cone from the IP to its own footprint on the
# barrel (its members' maximum Delta R from its axis), at a lower opacity than the jet's
# cone, and a thin outline on the cap.
set -eu
csv=$1; cand=$2; stem=$3; nsub=${4:-3}
cd "$(dirname "$0")"
awk -F, -v R=0.75 -v c="$cand" -v NSUB="$nsub" '
function x(eta, phi) { return R*cos(phi) }
function y(eta, phi) { return R*sin(phi) }
function z(eta, phi) { return R*(exp(eta)-exp(-eta))/2 }
function wrap(d) { while (d > pi) d -= 2*pi; while (d < -pi) d += 2*pi; return d }
function asinh(v) { return log(v + sqrt(v*v + 1)) }
function dr2(i, j) { return (CE[i]-CE[j])^2 + wrap(CP[i]-CP[j])^2 }
function kin(i) { CT[i]=sqrt(CX[i]^2+CY[i]^2); CE[i]=asinh(CZ[i]/CT[i]); CP[i]=atan2(CY[i],CX[i]) }
function cone(opac, lw, npts, EE, PP, k, k2) {
  printf "  \\begin{scope}[transparency group, opacity=%s]\n", opac
  for(k=0;k<npts;k++){ k2=(k+1)%npts; printf "    \\fill[jetcol,draw=jetcol,line width=%s,line join=round] (0,0,0) -- ({%.4f},{%.4f},{%.4f}) -- ({%.4f},{%.4f},{%.4f}) -- cycle;\n", lw, x(EE[k],PP[k]), y(EE[k],PP[k]), z(EE[k],PP[k]), x(EE[k2],PP[k2]), y(EE[k2],PP[k2]), z(EE[k2],PP[k2]) }
  printf "  \\end{scope}\n" }
function outline(style, npts, EE, PP, k) {
  printf "  \\path[%s] plot[smooth cycle,tension=0.7] coordinates {", style
  for(k=0;k<npts;k++) printf " ({%.4f},{%.4f},{%.4f})", x(EE[k],PP[k]), y(EE[k],PP[k]), z(EE[k],PP[k])
  printf " };\n" }
BEGIN { pi=3.14159265; RJET=0.8; FRAC=0.9 }
NR>1 && $1==c && $4=="none" {
  n++; PT[n]=$10; ETA[n]=$11; PHI[n]=$12; if($10>max)max=$10
  CX[n]=$7; CY[n]=$8; CZ[n]=$9; CT[n]=$10; CE[n]=$11; CP[n]=$12; ALIVE[n]=1; MEM[n]=n
  if(n==1){ aeta=$11-$13; aphi=$12-$14 }
}
END {
  # jet footprint
  dr=0
  for(i=1;i<=n;i++){ d=sqrt((ETA[i]-aeta)^2+wrap(PHI[i]-aphi)^2); if(d>dr)dr=d }
  dr+=0.04
  for(k=0;k<36;k++){ t=k*2*pi/36; FE[k]=aeta+dr*cos(t); FP[k]=aphi+dr*sin(t) }
  # exclusive kT down to NSUB subjets
  m=n
  while (m > NSUB) {
    best=-1
    for(i=1;i<=n;i++) if(ALIVE[i]) for(j=i+1;j<=n;j++) if(ALIVE[j]) {
      dij=(CT[i]<CT[j]?CT[i]^2:CT[j]^2)*dr2(i,j)/RJET^2
      if(best<0 || dij<best){ best=dij; bi=i; bj=j } }
    CX[bi]+=CX[bj]; CY[bi]+=CY[bj]; CZ[bi]+=CZ[bj]; kin(bi); MEM[bi]=MEM[bi] " " MEM[bj]; ALIVE[bj]=0; m--
  }
  printf "  %% jet cone and footprint, Delta R = %.2f about the axis (eta %.2f, phi %.2f rad)\n", dr, aeta, aphi
  cone("0.13", "0.3pt", 36, FE, FP)
  outline("fill=white,fill opacity=0.55", 36, FE, FP)
  outline("fill=jetcol,fill opacity=0.05,draw=jetcol,draw opacity=0.7,line width=0.4pt", 36, FE, FP)
  printf "  %% subjets from exclusive kT (R = %.1f, %d subjets): a smaller cone each\n", RJET, NSUB
  s=0
  for(i=1;i<=n;i++) if(ALIVE[i]) {
    s++; nm=split(MEM[i], MM, " "); tot=0
    for(q=1;q<=nm;q++){ DD[q]=sqrt((ETA[MM[q]]-CE[i])^2+wrap(PHI[MM[q]]-CP[i])^2); WW[q]=PT[MM[q]]; tot+=WW[q] }
    # radius holding FRAC of the subjet pT (insertion sort by Delta R, then cumulate)
    for(q=2;q<=nm;q++){ dq=DD[q]; wq=WW[q]; p=q-1; while(p>=1 && DD[p]>dq){ DD[p+1]=DD[p]; WW[p+1]=WW[p]; p-- } DD[p+1]=dq; WW[p+1]=wq }
    acc=0; sr=0; for(q=1;q<=nm;q++){ acc+=WW[q]; sr=DD[q]; if(acc>=FRAC*tot) break }
    sr+=0.02; if(sr<0.08) sr=0.08
    for(k=0;k<24;k++){ t=k*2*pi/24; SE[k]=CE[i]+sr*cos(t); SP[k]=CP[i]+sr*sin(t) }
    printf "  %% subjet %d: %d constituents, pT %.0f GeV, axis (eta %.2f, phi %.2f), Delta R %.2f\n", s, nm, CT[i], CE[i], CP[i], sr
    cone("0.10", "0.25pt", 24, SE, SP)
    outline("draw=jetcol,draw opacity=0.45,line width=0.3pt", 24, SE, SP)
    printf "subjet %d: %d constituents, pT %.0f GeV, dR %.2f\n", s, nm, CT[i], sr > "/dev/stderr"
  }
  printf "  %% constituents: discs in the barrel'"'"'s tangent plane at each position, area proportional to pT\n"
  for(i=1;i<=n;i++){ r=0.008+0.042*sqrt(PT[i]/max)
    cx=x(ETA[i],PHI[i]); cy=y(ETA[i],PHI[i]); cz=z(ETA[i],PHI[i]); ux=-sin(PHI[i]); uy=cos(PHI[i])
    printf "  \\fill[jetcol,opacity=0.7] plot[smooth cycle] coordinates {"
    for(k=0;k<12;k++){ t=k*2*pi/12; printf " ({%.4f},{%.4f},{%.4f})", cx+r*cos(t)*ux, cy+r*cos(t)*uy, cz+r*sin(t) }
    printf " };\n" }
  printf "axis eta %.3f phi %.1f deg, %d constituents, footprint dR %.2f\n", aeta, aphi*180/pi, n, dr > "/dev/stderr"
}' "$csv" > "jet_block_$stem.tex"
sed -e "s/@POINTS@/jet_block_$stem/" cylinder_template.tex > "cms_cylinder_$stem.tex"
pdflatex -interaction=nonstopmode "cms_cylinder_$stem.tex" > "cms_cylinder_$stem.compile.log" 2>&1 || true
grep -E "^!" "cms_cylinder_$stem.compile.log" | head -3 || true
pdftocairo -png -r 200 -singlefile "cms_cylinder_$stem.pdf" "cms_cylinder_$stem"
pdftocairo -png -r 600 -x 1250 -y 900 -W 750 -H 600 -singlefile "cms_cylinder_$stem.pdf" "cms_cylinder_${stem}_zoom"
echo "built cms_cylinder_$stem"
