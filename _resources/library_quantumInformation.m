(* __Functions__ *)

Fidelity[dm_, idealdm_] := Tr[MatrixPower[MatrixPower[idealdm, 1/2].dm.MatrixPower[idealdm, 1/2], 1/2]]^2 // Chop;

FidelityPure[x_, y_] := Sqrt[x\[ConjugateTranspose].y.x];
Purity[\[Rho]_]:=Tr[\[Rho].\[Rho]];

DensityMatrix:=#.#\[ConjugateTranspose]&
Kron:=KroneckerProduct[##]&
Kronk = Fold[KroneckerProduct];

neighborhoodLC = Exp[(I*\[Pi])/4] {{1, 0}, {0, -I}}
vertexLC = MatrixExp[(-I*\[Pi]/4)*(sx)]

StateMeasurementPure[state_,operator_]:=ConjugateTranspose@state.operator.state
StateMeasurement[rho_,operator_]:=Tr[operator.rho]

ProjectionMeasurement[x_, y_] := 
  FullSimplify[Abs[x\[ConjugateTranspose].y]^2][[1, 1]];

GetAngle[x_] := N@180*FullSimplify@ArcTan[x[[2]]/x[[1]]]/\[Pi];

EigenSolve[matrix_] := Block[{eigenvalues, eigenvectors},
  {eigenvalues, eigenvectors} = {#[[1]], #[[2]]} & /@ 
    Eigensystem[matrix];
  {eigenvalues, Normalize /@ eigenvectors}]

  GetProjectionAngles[
    proj_] :=
   {\[Theta] = 
     Chop@NMinimize[{Abs[(HWP[x].QWP[y].proj)[[2, 1]]], -\[Pi]/4 <= 
          x <= \[Pi]/4}, {x, y}][[2, {1, 2}, 2]],
    Chop@FullSimplify@((\[Theta]/\[Pi])*180)}

(*__BasisStates__*)

h = {{1}, {0}};
v = {{0}, {1}};
d = 1/Sqrt[2] (h + v);
a = 1/Sqrt[2] (h - v);
r = 1/Sqrt[2] (h + I*v);
l = 1/Sqrt[2] (h - I*v);

hh = Kron[h, h]; hv = Kron[h, v]; hd = Kron[h, d];
ha = Kron[h, a]; hr = Kron[h, r]; hl = Kron[h, l];
vh = Kron[v, h]; vv = Kron[v, v]; vd = Kron[v, d];
va = Kron[v, a]; vr = Kron[v, r]; vl = Kron[v, l];
dh = Kron[d, h]; dv = Kron[d, v]; dd = Kron[d, d];
da = Kron[d, a]; dr = Kron[d, r]; dl = Kron[d, l];
ah = Kron[a, h]; av = Kron[a, v]; ad = Kron[a, v];
aa = Kron[a, a]; ar = Kron[a, r]; al = Kron[a, l];
rh = Kron[r, h]; rv = Kron[r, v]; rd = Kron[r, v];
ra = Kron[r, a]; rr = Kron[r, r]; rl = Kron[r, l];
lh = Kron[l, h]; lv = Kron[l, v]; ld = Kron[l, v];
la = Kron[l, a]; lr = Kron[l, r]; ll = Kron[l, l];

s0 = {{1, 0}, {0, 1}};
sx = {{0, 1}, {1, 0}};
sy = {{0, -I}, {I, 0}};
sz = {{1, 0}, {0, -1}}; 
Hada = 1/Sqrt[2] {{1, 1}, {1, -1}};

GHZ[nQubit_]:=1/Sqrt[2] (Kron@@ConstantArray[h,nQubit]+Kron@@ConstantArray[v,nQubit])/;nQubit>=2

(* __Gates__ *)

cNot = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}};
cH = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1/\[Sqrt]2, 1/\[Sqrt]2}, {0, 0, 1/\[Sqrt]2, -(1/\[Sqrt]2)}};
cX = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}};
cY = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, -I}, {0, 0, I, 0}};
cZ = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, -1}};
cI = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
swap = {{1, 0, 0, 0}, {0, 0, 1, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}};

(* __MoreFucntions__ *)
 
BitFlip[p_, \[Rho]_] := p.cX.\[Rho].cX + (1 - p) \[Rho];
PhaseFlip[p_, \[Rho]_] := p*sz.\[Rho].sz + (1 - p) \[Rho];
BitPhaseFlip[p_, \[Rho]_] := p*sy.\[Rho].sy + (1 - p) \[Rho];
 
DepolControl[p_, \[Rho]_] := (1 - 3/4*p) \[Rho] + p/4*(cX.\[Rho].cX + cY.\[Rho].cY + cZ.\[Rho].cZ);
 
Depol[\[Eta]_, \[Rho]_] := (1 - \[Eta]) \[Rho] + \[Eta] idn/2;
 
Depol1[p_, \[Rho]_] := (1 - 3/4*p) \[Rho] + p/4*(sx.\[Rho].sx + sy.\[Rho].Y + sz.\[Rho].sz);
 
Depol2[p_, \[Rho]_] := p/2*cI + (1 - p) \[Rho];
 
(* Dephase[{\[Alpha]x_, \[Alpha]y_, \[Alpha]z_}, p_, \[Rho]_] := (1 - p/2) \[Rho] + p/2 (\[Alpha]x sx.\[Rho].sx + \[Alpha]y sy.\[Rho].sy + \[Alpha]z \sz.\[Rho].sz); *)
 
DepolMemory[\[Eta]_, \[Rho]_] := \[Eta] \[Rho] + (1 - \[Eta]) idn/2;


QWP[t_]:=1/Sqrt[2] ({
 {1+I Cos[2t], I Sin[2t]},
 {I Sin[2t], 1-I Cos[2t]}
});

HWP[t_]:=Exp[(I \[Pi])/2]({
 {Cos[2t], Sin[2t]},
 {Sin[2t], -Cos[2t]}
});

(* __PlottingFunctions__ *)

Block[{t0, shift, lt, st, alpha0}, lt = 0.015;
 st = 0.01;
 dmticks = 
  Join[{#, #, {lt, 0}} & /@ Range[-1, 1, 1/4], {#, "", {st, 0}} & /@ 
    Range[-1, 1, 1/16]];]


Charting`iBarChart3D[{}];
prot = Unprotect@Charting`iBarChart3D;
DownValues@Charting`iBarChart3D = 
  DownValues@Charting`iBarChart3D /. 
   HoldPattern[
     lhs : _["ChartBaseStyle"] = 
      cd_[If[cond_, then_, else_], rest__]] :> (If[cond, 
      lhs = cd[rest];
      If[! MemberQ[lhs, _EdgeForm], AppendTo[lhs, then]], 
      lhs = cd[else, rest]]);
Protect /@ prot;

Unprotect[ColorData]
ColorData["BlueGreenYellow"] = 
  Function[x, 
   Blend[{RGBColor[0.49, 0.49, 0.49, 0.3], 
     RGBColor[{0.26700401, 0.00487433, 0.32941519}], 
     RGBColor[{0.26851048, 0.00960483, 0.33542652}], 
     RGBColor[{0.26994384, 0.01462494, 0.34137895}], 
     RGBColor[{0.27130489, 0.01994186, 0.34726862}], 
     RGBColor[{0.27259384, 0.02556309, 0.35309303}], 
     RGBColor[{0.27380934, 0.03149748, 0.35885256}], 
     RGBColor[{0.27495242, 0.03775181, 0.36454323}], 
     RGBColor[{0.27602238, 0.04416723, 0.37016418}], 
     RGBColor[{0.2770184, 0.05034437, 0.37571452}], 
     RGBColor[{0.27794143, 0.05632444, 0.38119074}], 
     RGBColor[{0.27879067, 0.06214536, 0.38659204}], 
     RGBColor[{0.2795655, 0.06783587, 0.39191723}], 
     RGBColor[{0.28026658, 0.07341724, 0.39716349}], 
     RGBColor[{0.28089358, 0.07890703, 0.40232944}], 
     RGBColor[{0.28144581, 0.0843197, 0.40741404}], 
     RGBColor[{0.28192358, 0.08966622, 0.41241521}], 
     RGBColor[{0.28232739, 0.09495545, 0.41733086}], 
     RGBColor[{0.28265633, 0.10019576, 0.42216032}], 
     RGBColor[{0.28291049, 0.10539345, 0.42690202}], 
     RGBColor[{0.28309095, 0.11055307, 0.43155375}], 
     RGBColor[{0.28319704, 0.11567966, 0.43611482}], 
     RGBColor[{0.28322882, 0.12077701, 0.44058404}], 
     RGBColor[{0.28318684, 0.12584799, 0.44496}], 
     RGBColor[{0.283072, 0.13089477, 0.44924127}], 
     RGBColor[{0.28288389, 0.13592005, 0.45342734}], 
     RGBColor[{0.28262297, 0.14092556, 0.45751726}], 
     RGBColor[{0.28229037, 0.14591233, 0.46150995}], 
     RGBColor[{0.28188676, 0.15088147, 0.46540474}], 
     RGBColor[{0.28141228, 0.15583425, 0.46920128}], 
     RGBColor[{0.28086773, 0.16077132, 0.47289909}], 
     RGBColor[{0.28025468, 0.16569272, 0.47649762}], 
     RGBColor[{0.27957399, 0.17059884, 0.47999675}], 
     RGBColor[{0.27882618, 0.1754902, 0.48339654}], 
     RGBColor[{0.27801236, 0.18036684, 0.48669702}], 
     RGBColor[{0.27713437, 0.18522836, 0.48989831}], 
     RGBColor[{0.27619376, 0.19007447, 0.49300074}], 
     RGBColor[{0.27519116, 0.1949054, 0.49600488}], 
     RGBColor[{0.27412802, 0.19972086, 0.49891131}], 
     RGBColor[{0.27300596, 0.20452049, 0.50172076}], 
     RGBColor[{0.27182812, 0.20930306, 0.50443413}], 
     RGBColor[{0.27059473, 0.21406899, 0.50705243}], 
     RGBColor[{0.26930756, 0.21881782, 0.50957678}], 
     RGBColor[{0.26796846, 0.22354911, 0.5120084}], 
     RGBColor[{0.26657984, 0.2282621, 0.5143487}], 
     RGBColor[{0.2651445, 0.23295593, 0.5165993}], 
     RGBColor[{0.2636632, 0.23763078, 0.51876163}], 
     RGBColor[{0.26213801, 0.24228619, 0.52083736}], 
     RGBColor[{0.26057103, 0.2469217, 0.52282822}], 
     RGBColor[{0.25896451, 0.25153685, 0.52473609}], 
     RGBColor[{0.25732244, 0.2561304, 0.52656332}], 
     RGBColor[{0.25564519, 0.26070284, 0.52831152}], 
     RGBColor[{0.25393498, 0.26525384, 0.52998273}], 
     RGBColor[{0.25219404, 0.26978306, 0.53157905}], 
     RGBColor[{0.25042462, 0.27429024, 0.53310261}], 
     RGBColor[{0.24862899, 0.27877509, 0.53455561}], 
     RGBColor[{0.2468114, 0.28323662, 0.53594093}], 
     RGBColor[{0.24497208, 0.28767547, 0.53726018}], 
     RGBColor[{0.24311324, 0.29209154, 0.53851561}], 
     RGBColor[{0.24123708, 0.29648471, 0.53970946}], 
     RGBColor[{0.23934575, 0.30085494, 0.54084398}], 
     RGBColor[{0.23744138, 0.30520222, 0.5419214}], 
     RGBColor[{0.23552606, 0.30952657, 0.54294396}], 
     RGBColor[{0.23360277, 0.31382773, 0.54391424}], 
     RGBColor[{0.2316735, 0.3181058, 0.54483444}], 
     RGBColor[{0.22973926, 0.32236127, 0.54570633}], 
     RGBColor[{0.22780192, 0.32659432, 0.546532}], 
     RGBColor[{0.2258633, 0.33080515, 0.54731353}], 
     RGBColor[{0.22392515, 0.334994, 0.54805291}], 
     RGBColor[{0.22198915, 0.33916114, 0.54875211}], 
     RGBColor[{0.22005691, 0.34330688, 0.54941304}], 
     RGBColor[{0.21812995, 0.34743154, 0.55003755}], 
     RGBColor[{0.21620971, 0.35153548, 0.55062743}], 
     RGBColor[{0.21429757, 0.35561907, 0.5511844}], 
     RGBColor[{0.21239477, 0.35968273, 0.55171011}], 
     RGBColor[{0.2105031, 0.36372671, 0.55220646}], 
     RGBColor[{0.20862342, 0.36775151, 0.55267486}], 
     RGBColor[{0.20675628, 0.37175775, 0.55311653}], 
     RGBColor[{0.20490257, 0.37574589, 0.55353282}], 
     RGBColor[{0.20306309, 0.37971644, 0.55392505}], 
     RGBColor[{0.20123854, 0.38366989, 0.55429441}], 
     RGBColor[{0.1994295, 0.38760678, 0.55464205}], 
     RGBColor[{0.1976365, 0.39152762, 0.55496905}], 
     RGBColor[{0.19585993, 0.39543297, 0.55527637}], 
     RGBColor[{0.19410009, 0.39932336, 0.55556494}], 
     RGBColor[{0.19235719, 0.40319934, 0.55583559}], 
     RGBColor[{0.19063135, 0.40706148, 0.55608907}], 
     RGBColor[{0.18892259, 0.41091033, 0.55632606}], 
     RGBColor[{0.18723083, 0.41474645, 0.55654717}], 
     RGBColor[{0.18555593, 0.4185704, 0.55675292}], 
     RGBColor[{0.18389763, 0.42238275, 0.55694377}], 
     RGBColor[{0.18225561, 0.42618405, 0.5571201}], 
     RGBColor[{0.18062949, 0.42997486, 0.55728221}], 
     RGBColor[{0.17901879, 0.43375572, 0.55743035}], 
     RGBColor[{0.17742298, 0.4375272, 0.55756466}], 
     RGBColor[{0.17584148, 0.44128981, 0.55768526}], 
     RGBColor[{0.17427363, 0.4450441, 0.55779216}], 
     RGBColor[{0.17271876, 0.4487906, 0.55788532}], 
     RGBColor[{0.17117615, 0.4525298, 0.55796464}], 
     RGBColor[{0.16964573, 0.45626209, 0.55803034}], 
     RGBColor[{0.16812641, 0.45998802, 0.55808199}], 
     RGBColor[{0.1666171, 0.46370813, 0.55811913}], 
     RGBColor[{0.16511703, 0.4674229, 0.55814141}], 
     RGBColor[{0.16362543, 0.47113278, 0.55814842}], 
     RGBColor[{0.16214155, 0.47483821, 0.55813967}], 
     RGBColor[{0.16066467, 0.47853961, 0.55811466}], 
     RGBColor[{0.15919413, 0.4822374, 0.5580728}], 
     RGBColor[{0.15772933, 0.48593197, 0.55801347}], 
     RGBColor[{0.15626973, 0.4896237, 0.557936}], 
     RGBColor[{0.15481488, 0.49331293, 0.55783967}], 
     RGBColor[{0.15336445, 0.49700003, 0.55772371}], 
     RGBColor[{0.1519182, 0.50068529, 0.55758733}], 
     RGBColor[{0.15047605, 0.50436904, 0.55742968}], 
     RGBColor[{0.14903918, 0.50805136, 0.5572505}], 
     RGBColor[{0.14760731, 0.51173263, 0.55704861}], 
     RGBColor[{0.14618026, 0.51541316, 0.55682271}], 
     RGBColor[{0.14475863, 0.51909319, 0.55657181}], 
     RGBColor[{0.14334327, 0.52277292, 0.55629491}], 
     RGBColor[{0.14193527, 0.52645254, 0.55599097}], 
     RGBColor[{0.14053599, 0.53013219, 0.55565893}], 
     RGBColor[{0.13914708, 0.53381201, 0.55529773}], 
     RGBColor[{0.13777048, 0.53749213, 0.55490625}], 
     RGBColor[{0.1364085, 0.54117264, 0.55448339}], 
     RGBColor[{0.13506561, 0.54485335, 0.55402906}], 
     RGBColor[{0.13374299, 0.54853458, 0.55354108}], 
     RGBColor[{0.13244401, 0.55221637, 0.55301828}], 
     RGBColor[{0.13117249, 0.55589872, 0.55245948}], 
     RGBColor[{0.1299327, 0.55958162, 0.55186354}], 
     RGBColor[{0.12872938, 0.56326503, 0.55122927}], 
     RGBColor[{0.12756771, 0.56694891, 0.55055551}], 
     RGBColor[{0.12645338, 0.57063316, 0.5498411}], 
     RGBColor[{0.12539383, 0.57431754, 0.54908564}], 
     RGBColor[{0.12439474, 0.57800205, 0.5482874}], 
     RGBColor[{0.12346281, 0.58168661, 0.54744498}], 
     RGBColor[{0.12260562, 0.58537105, 0.54655722}], 
     RGBColor[{0.12183122, 0.58905521, 0.54562298}], 
     RGBColor[{0.12114807, 0.59273889, 0.54464114}], 
     RGBColor[{0.12056501, 0.59642187, 0.54361058}], 
     RGBColor[{0.12009154, 0.60010387, 0.54253043}], 
     RGBColor[{0.11973756, 0.60378459, 0.54139999}], 
     RGBColor[{0.11951163, 0.60746388, 0.54021751}], 
     RGBColor[{0.11942341, 0.61114146, 0.53898192}], 
     RGBColor[{0.11948255, 0.61481702, 0.53769219}], 
     RGBColor[{0.11969858, 0.61849025, 0.53634733}], 
     RGBColor[{0.12008079, 0.62216081, 0.53494633}], 
     RGBColor[{0.12063824, 0.62582833, 0.53348834}], 
     RGBColor[{0.12137972, 0.62949242, 0.53197275}], 
     RGBColor[{0.12231244, 0.63315277, 0.53039808}], 
     RGBColor[{0.12344358, 0.63680899, 0.52876343}], 
     RGBColor[{0.12477953, 0.64046069, 0.52706792}], 
     RGBColor[{0.12632581, 0.64410744, 0.52531069}], 
     RGBColor[{0.12808703, 0.64774881, 0.52349092}], 
     RGBColor[{0.13006688, 0.65138436, 0.52160791}], 
     RGBColor[{0.13226797, 0.65501363, 0.51966086}], 
     RGBColor[{0.13469183, 0.65863619, 0.5176488}], 
     RGBColor[{0.13733921, 0.66225157, 0.51557101}], 
     RGBColor[{0.14020991, 0.66585927, 0.5134268}], 
     RGBColor[{0.14330291, 0.66945881, 0.51121549}], 
     RGBColor[{0.1466164, 0.67304968, 0.50893644}], 
     RGBColor[{0.15014782, 0.67663139, 0.5065889}], 
     RGBColor[{0.15389405, 0.68020343, 0.50417217}], 
     RGBColor[{0.15785146, 0.68376525, 0.50168574}], 
     RGBColor[{0.16201598, 0.68731632, 0.49912906}], 
     RGBColor[{0.1663832, 0.69085611, 0.49650163}], 
     RGBColor[{0.1709484, 0.69438405, 0.49380294}], 
     RGBColor[{0.17570671, 0.6978996, 0.49103252}], 
     RGBColor[{0.18065314, 0.70140222, 0.48818938}], 
     RGBColor[{0.18578266, 0.70489133, 0.48527326}], 
     RGBColor[{0.19109018, 0.70836635, 0.48228395}], 
     RGBColor[{0.19657063, 0.71182668, 0.47922108}], 
     RGBColor[{0.20221902, 0.71527175, 0.47608431}], 
     RGBColor[{0.20803045, 0.71870095, 0.4728733}], 
     RGBColor[{0.21400015, 0.72211371, 0.46958774}], 
     RGBColor[{0.22012381, 0.72550945, 0.46622638}], 
     RGBColor[{0.2263969, 0.72888753, 0.46278934}], 
     RGBColor[{0.23281498, 0.73224735, 0.45927675}], 
     RGBColor[{0.2393739, 0.73558828, 0.45568838}], 
     RGBColor[{0.24606968, 0.73890972, 0.45202405}], 
     RGBColor[{0.25289851, 0.74221104, 0.44828355}], 
     RGBColor[{0.25985676, 0.74549162, 0.44446673}], 
     RGBColor[{0.26694127, 0.74875084, 0.44057284}], 
     RGBColor[{0.27414922, 0.75198807, 0.4366009}], 
     RGBColor[{0.28147681, 0.75520266, 0.43255207}], 
     RGBColor[{0.28892102, 0.75839399, 0.42842626}], 
     RGBColor[{0.29647899, 0.76156142, 0.42422341}], 
     RGBColor[{0.30414796, 0.76470433, 0.41994346}], 
     RGBColor[{0.31192534, 0.76782207, 0.41558638}], 
     RGBColor[{0.3198086, 0.77091403, 0.41115215}], 
     RGBColor[{0.3277958, 0.77397953, 0.40664011}], 
     RGBColor[{0.33588539, 0.7770179, 0.40204917}], 
     RGBColor[{0.34407411, 0.78002855, 0.39738103}], 
     RGBColor[{0.35235985, 0.78301086, 0.39263579}], 
     RGBColor[{0.36074053, 0.78596419, 0.38781353}], 
     RGBColor[{0.3692142, 0.78888793, 0.38291438}], 
     RGBColor[{0.37777892, 0.79178146, 0.3779385}], 
     RGBColor[{0.38643282, 0.79464415, 0.37288606}], 
     RGBColor[{0.39517408, 0.79747541, 0.36775726}], 
     RGBColor[{0.40400101, 0.80027461, 0.36255223}], 
     RGBColor[{0.4129135, 0.80304099, 0.35726893}], 
     RGBColor[{0.42190813, 0.80577412, 0.35191009}], 
     RGBColor[{0.43098317, 0.80847343, 0.34647607}], 
     RGBColor[{0.44013691, 0.81113836, 0.3409673}], 
     RGBColor[{0.44936763, 0.81376835, 0.33538426}], 
     RGBColor[{0.45867362, 0.81636288, 0.32972749}], 
     RGBColor[{0.46805314, 0.81892143, 0.32399761}], 
     RGBColor[{0.47750446, 0.82144351, 0.31819529}], 
     RGBColor[{0.4870258, 0.82392862, 0.31232133}], 
     RGBColor[{0.49661536, 0.82637633, 0.30637661}], 
     RGBColor[{0.5062713, 0.82878621, 0.30036211}], 
     RGBColor[{0.51599182, 0.83115784, 0.29427888}], 
     RGBColor[{0.52577622, 0.83349064, 0.2881265}], 
     RGBColor[{0.5356211, 0.83578452, 0.28190832}], 
     RGBColor[{0.5455244, 0.83803918, 0.27562602}], 
     RGBColor[{0.55548397, 0.84025437, 0.26928147}], 
     RGBColor[{0.5654976, 0.8424299, 0.26287683}], 
     RGBColor[{0.57556297, 0.84456561, 0.25641457}], 
     RGBColor[{0.58567772, 0.84666139, 0.24989748}], 
     RGBColor[{0.59583934, 0.84871722, 0.24332878}], 
     RGBColor[{0.60604528, 0.8507331, 0.23671214}], 
     RGBColor[{0.61629283, 0.85270912, 0.23005179}], 
     RGBColor[{0.62657923, 0.85464543, 0.22335258}], 
     RGBColor[{0.63690157, 0.85654226, 0.21662012}], 
     RGBColor[{0.64725685, 0.85839991, 0.20986086}], 
     RGBColor[{0.65764197, 0.86021878, 0.20308229}], 
     RGBColor[{0.66805369, 0.86199932, 0.19629307}], 
     RGBColor[{0.67848868, 0.86374211, 0.18950326}], 
     RGBColor[{0.68894351, 0.86544779, 0.18272455}], 
     RGBColor[{0.69941463, 0.86711711, 0.17597055}], 
     RGBColor[{0.70989842, 0.86875092, 0.16925712}], 
     RGBColor[{0.72039115, 0.87035015, 0.16260273}], 
     RGBColor[{0.73088902, 0.87191584, 0.15602894}], 
     RGBColor[{0.74138803, 0.87344918, 0.14956101}], 
     RGBColor[{0.75188414, 0.87495143, 0.14322828}], 
     RGBColor[{0.76237342, 0.87642392, 0.13706449}], 
     RGBColor[{0.77285183, 0.87786808, 0.13110864}], 
     RGBColor[{0.78331535, 0.87928545, 0.12540538}], 
     RGBColor[{0.79375994, 0.88067763, 0.12000532}], 
     RGBColor[{0.80418159, 0.88204632, 0.11496505}], 
     RGBColor[{0.81457634, 0.88339329, 0.11034678}], 
     RGBColor[{0.82494028, 0.88472036, 0.10621724}], 
     RGBColor[{0.83526959, 0.88602943, 0.1026459}], 
     RGBColor[{0.84556056, 0.88732243, 0.09970219}], 
     RGBColor[{0.8558096, 0.88860134, 0.09745186}], 
     RGBColor[{0.86601325, 0.88986815, 0.09595277}], 
     RGBColor[{0.87616824, 0.89112487, 0.09525046}], 
     RGBColor[{0.88627146, 0.89237353, 0.09537439}], 
     RGBColor[{0.89632002, 0.89361614, 0.09633538}], 
     RGBColor[{0.90631121, 0.89485467, 0.09812496}], 
     RGBColor[{0.91624212, 0.89609127, 0.1007168}], 
     RGBColor[{0.92610579, 0.89732977, 0.10407067}], 
     RGBColor[{0.93590444, 0.8985704, 0.10813094}], 
     RGBColor[{0.94563626, 0.899815, 0.11283773}], 
     RGBColor[{0.95529972, 0.90106534, 0.11812832}], 
     RGBColor[{0.96489353, 0.90232311, 0.12394051}], 
     RGBColor[{0.97441665, 0.90358991, 0.13021494}], 
     RGBColor[{0.98386829, 0.90486726, 0.13689671}], 
     RGBColor[{0.99324789, 0.90615657, 0.1439362}]}, x]];
Protect[ColorData]

RGBColorConversion[{a_, b_, c_}] := RGBColor[a/256, b/256, c/256]
rocketRGBvalues = {{37, 25, 56}, {81, 45, 86}, {128, 57, 104}, {169, 
    59, 108}, {206, 73, 94}, {228, 119, 90}, {241, 199, 170}, {248, 
    233, 220}};
rocketColors = RGBColorConversion /@ rocketRGBvalues

Unprotect[ColorData]
  ColorData["AuroraColors"] = 
    Blend[rocketColors, #*1] & ;
Protect[ColorData]
  
DensityMatrixLabels[nQubits_] := Tuples[{"H", "V", "D", "A", "R", "L"}, nQubits];
DensityMatrixPlotLabels[nQubits_] := Tuples[{"H", "V"}, nQubits];

font = FontFamily -> "Times New Roman";
fontSize = 18;

(*DensityMatrixPlot[densitymatrx_] := 
 Show[BarChart3D[Re[densitymatrx], ChartLayout -> "Grid", 
   ChartElementFunction -> 
    ChartElementDataFunction["GradientScaleCube", 
     "ColorScheme" -> "DeepSeaColors"], BarSpacing -> {0.2, .2}, 
   BaseStyle -> {FontFamily -> font, FontSize -> fontSize, Black}, 
   Method -> {"Canvas" -> None}, 
   ChartBaseStyle -> EdgeForm[{Thickness[.001], Opacity[1], Black}], 
   Ticks -> {None, None, dmticks}, Boxed -> True, 
   BoxStyle -> {Directive[Thickness[0.002], Opacity[.6], Gray, 
      Dashed]}, FaceGrids -> None, Axes -> {False, False, True}, 
   AxesStyle -> {Directive[Black, Opacity[1]], 
     Directive[Black, Opacity[1]], Directive[Black]}, 
   BoxRatios -> {1, 1, 1.5}, ImageSize -> Large, 
   AspectRatio -> 1/GoldenRatio, ViewPoint -> {1.3, -2.4, 2.}]]*)

DensityMatrixPlot[densitymatrx_] := Show[BarChart3D[Re[densitymatrx],
   ChartLayout -> "Grid",
   ChartElementFunction -> 
    ChartElementDataFunction["GradientScaleCube", 
     "ColorScheme" -> "DeepSeaColors"],
   BarSpacing -> {.2, .2},
   BaseStyle -> {FontFamily -> font, FontSize -> fontSize, Black},
   Method -> {"Canvas" -> None},
   ChartBaseStyle -> EdgeForm[{Thickness[.001], Opacity[1], Gray}],
   Ticks -> {Automatic, Automatic, dmticks},
   Boxed -> True, 
   BoxStyle -> {Directive[Thickness[0.001], Opacity[1], Gray, Dashed]},
   FaceGrids -> None,
   Axes -> {True, True, True},
   AxesStyle -> {Directive[Black, Opacity[0]], 
     Directive[Black, Opacity[0]], Directive[Black]},
   BoxRatios -> {1, 1, .6},
   ImageSize -> Large,
   ViewPoint -> {1.3, -2.4, 2.},
   LabelStyle -> Directive[FontFamily -> "YuMincho", FontSize -> 12]
   ]
  ]

DensityMatrixPlotFull[densitymatrx_] := 
 GraphicsGrid[{{BarChart3D[Re[densitymatrx]
     , ChartLayout -> "Grid"
     , ChartElementFunction -> 
      ChartElementDataFunction["GradientScaleCube", 
       "ColorScheme" -> "GrayYellowTones"]
     , BarSpacing -> {0, 0}
     , BaseStyle -> {FontFamily -> font, FontSize -> fontSize, Black}
     , Method -> {"Canvas" -> None}
     , ChartBaseStyle -> EdgeForm[{Thickness[.002], Opacity[1], Black}]
     , Ticks -> {None, None, dmticks}
     , Boxed -> True, 
     BoxStyle -> {Directive[Thickness[0.002], Opacity[.6], Gray, 
        Dashed]}
     , FaceGrids -> None
     , Axes -> {False, False, True}
     , AxesStyle -> {Directive[Black, Opacity[1]], 
       Directive[Black, Opacity[1]], Directive[Black]}
     , BoxRatios -> {.1, .1, 1}
     , ImageSize -> Large
     , AspectRatio -> 0.8
     , ViewPoint -> {1.3, -2.4, 2.}
     ],
    
    BarChart3D[Im[densitymatrx]
     , ChartLayout -> "Grid"
     , ChartElementFunction -> 
      ChartElementDataFunction["GradientScaleCube", 
       "ColorScheme" -> "GrayYellowTones"]
     , BarSpacing -> {0, 0}
     , BaseStyle -> {FontFamily -> font, FontSize -> fontSize, Black}
     , Method -> {"Canvas" -> None}
     , ChartBaseStyle -> EdgeForm[{Thickness[.002], Opacity[1], Black}]
     , Ticks -> {None, None, dmticks}
     , Boxed -> True, 
     BoxStyle -> {Directive[Thickness[0.002], Opacity[.6], Gray, 
        Dashed]}
     , FaceGrids -> None
     , Axes -> {False, False, True}
     , AxesStyle -> {Directive[Black, Opacity[1]], 
       Directive[Black, Opacity[1]], Directive[Black]}
     , BoxRatios -> {.1, .1, 1}
     , ImageSize -> Large
     , AspectRatio -> 0.8
     , ViewPoint -> {1.3, -2.4, 2.}
     ]}}]
     SwapParts[expr_, pos1_, pos2_] := 
     ReplacePart[#, #, {pos1, pos2}, {pos2, pos1}] &[expr];
   
TraceSystem[dm_, sys_] := 
 Block[{Qubits, TrkM, n, M, k, Permut, perm, b, c, p},
  Qubits = Reverse[Sort[sys]];(* 
  rearrange the list of qubits to be traced out *)
  TrkM = dm;
  
  (* For all qubits to be traced out... *)
  
  For[q = 1, q <= Dimensions[Qubits][[1]], q++,
   n = Log[2, (Dimensions[TrkM][[1]])]; (* 
   dimensions of original system *)
   M = TrkM;
   k = Qubits[[q]];
   
   If[k == n,(* if tracing the last system *)
    TrkM = {};
    For[p = 1, p < 2^n + 1, p = p + 2,
     TrkM = 
      Append[TrkM, 
       Take[M[[p, All]], {1, 2^n, 2}] + 
        Take[M[[p + 1, All]], {2, 2^n, 2}]];(* Pick row p/
     p+1 and take elements 1/2 through 2^n in steps of 2 - 
     sum thoese and append to zero matrix - 
     do for all rows *)
      ],
    
    For[j = 0, j < (n - k), j++,(* if not - permute accordingly *)
 
         b = {0};
     For[i = 1, i < 2^n + 1, i++,
      If[(Mod[(IntegerDigits[i - 1, 2, n][[n]] + 
             IntegerDigits[i - 1, 2, n][[n - j - 1]]), 2]) == 1 && 
        Count[b, i]  == 0, 
       Permut = {i, (FromDigits[
            SwapParts[(IntegerDigits[i - 1, 2, n]), {n}, {n - j - 
               1}], 2] + 1)};
       b = 
        Append[b, (FromDigits[
            SwapParts[(IntegerDigits[i - 1, 2, n]), {n}, {n - j - 
               1}], 2] + 1)];
       c = Range[2^n];
       perm = 
        SwapParts[
         c, {i}, {(FromDigits[
             SwapParts[(IntegerDigits[i - 1, 2, n]), {n}, {n - j - 
                1}], 2] + 1)}];
       
       M = M[[perm, perm]];
       
        ]    
      ];
     (* and now trace out last system *)
     TrkM = {};
     For[p = 1, p < 2^n + 1, p = p + 2,
      TrkM = 
        Append[TrkM, 
         Take[M[[p, All]], {1, 2^n, 2}] + 
          Take[M[[p + 1, All]], {2, 2^n, 2}]];
      ]
        ]
    ];
   
   ];
  Clear[Qubits, n, M, k, Permut, perm, b, c];
  TrkM
  ];

Coding[n_] :=
 
 "|" <> # <> "\[RightAngleBracket]" & /@ 
  Table[IntegerString[i - 1, 2, n], {i, 1, 2^n}];

DensityMatrixPlot[densitymatrx_] := Show[BarChart3D[Re[densitymatrx],

ChartLayout -> "Grid",
ChartElementFunction -> 
ChartElementDataFunction["GradientScaleCube", 
 "ColorScheme" -> "AuroraColors"],
BarSpacing -> {.2, .2},
BaseStyle -> {FontFamily -> font, FontSize -> fontSize, Black},
Method -> {"Canvas" -> None},
ChartBaseStyle -> EdgeForm[{Thickness[.001], Opacity[1], Gray}],
Ticks -> {Automatic, Automatic, dmticks},
Boxed -> True, 
BoxStyle -> {Directive[Thickness[0.001], Opacity[1], Gray, Dashed]},
FaceGrids -> None,
Axes -> {True, True, True},
AxesStyle -> {Directive[Black, Opacity[0]], 
 Directive[Black, Opacity[0]], Directive[Black]},
BoxRatios -> {1, 1, .6},
ImageSize -> Large,
ViewPoint -> {1.3, -2.4, 2.},

ChartLabels -> {{""},
 Placed[labels, Axis, Rotate[#, -30*\[Pi]/180] &],
 Placed[labels, Axis, Rotate[#, Pi/4] &]},
LabelStyle -> Directive[FontFamily -> "YuMincho", FontSize -> 12]
]
]