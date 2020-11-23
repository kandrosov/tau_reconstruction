#include <iostream>
#include "TLorentzVector.h"
#include "Math/Vector4D.h"

using namespace ROOT::Math;

TLorentzVector *v = new TLorentzVector(); 
v.SetPtEtaPhiM(pt, eta,phi,m);
float x1,x2;

x1=v.Pt();
x2=v[0];

std::cout << x1 << std::endl;