Mesh.MshFileVersion = 2.0;
// Cylinder parameters
// Units in micrometer
D = 11.2;
r = D/2;
Lstart = 0;
Lend = 5*r;
L = Lend-Lstart; 
h = 0.5;

// Points and extrude
Point(1) = {0,  0, Lstart, h};
Point(2) = {r,  0, Lstart, h};
Point(3) = {0,  r, Lstart, h};
Point(4) = {-r, 0, Lstart, h};
Point(5) = {0, -r, Lstart, h};

/* Point(1) = {0,  0, Lstart}; */
/* Point(2) = {r,  0, Lstart}; */
/* Point(3) = {0,  r, Lstart}; */
/* Point(4) = {-r, 0, Lstart}; */
/* Point(5) = {0, -r, Lstart}; */

Circle(1) = {2,1,3};
Circle(2) = {3,1,4};
Circle(3) = {4,1,5};
Circle(4) = {5,1,2};
Line Loop(5) = {1,2,3,4};
Plane Surface(6) = {5};

Extrude {0,0,L} {
  Surface{6};
}


// Boundary and Subdomain tags
Physical Surface("1") = {23, 19, 15, 27};
Physical Surface("2") = {6};
Physical Surface("3") = {28};
Physical Volume("1") = {1};

/* Field[1] = Box; */
/* refine_r = 0.5*r; */
/* h_fine = r/20; */
/* h_coarse = r/5; */
/* Field[1].Thickness = 0.1*refine_r; */
/* Field[1].VIn = h_coarse; */
/* Field[1].VOut = h_fine; */
/* Field[1].XMax = refine_r; */
/* Field[1].XMin = -refine_r; */
/* Field[1].YMax = refine_r; */
/* Field[1].YMin = -refine_r; */
/* Field[1].ZMax = Lend; */
/* Field[1].ZMin = Lstart; */
/* Background Field = 1; */
 
/* Field[1] = BoundaryLayer; */
/* Field[1].AnisoMax = 1000; */
/* Field[1].Quads = 0; */
/* Field[1].Thickness = 5; */
/* Field[1].CurvesList = {6, 28}; */
/* Field[1].Thickness = 3; */
/* Field[1].NbLayers = 10; */
/* Field[1].Quads = 0; */
/* Field[1].Ratio = 1.1; */
/* Field[1].Size = h/4; */
/* Field[1].SizeFar = h; */
/* BoundaryLayer Field = 1; */

/* //+ */
/* Field[1] = MathEval; */
/* Field[1].F = "0.4"; */
/* Field[2] = Restrict; */
/* /1* Field[2].IncludeBoundary = 1; *1/ */
/* Field[2].SurfacesList = {23, 19}; */
/* Background Field = 2; */
