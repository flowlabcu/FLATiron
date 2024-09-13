Mesh.MshFileVersion = 2.0;

/*

1-------------------2
|   |   |   |   |   |
|   |   |   |   |   |
4-------------------3



*/

// -- Parameters

w  = 1;
h  = 1;
dx = 1e-2;

// -- Points
Point(1) = {0, h, 0, dx};
Point(2) = {w, h, 0, dx};
Point(3) = {w, 0, 0, dx};
Point(4) = {0, 0, 0, dx};

// -- Lines
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

/* Field[1] = Box; */
/* Field[1].VIn = 0.01; */
/* Field[1].VOut = dx; */
/* Field[1].XMax = 1.2; */
/* Field[1].XMin = 0; */
/* Field[1].YMax = 4.7; */
/* Field[1].YMin = 3; */
/* Field[1].ZMax = 0; */
/* Field[1].ZMin = 0; */
/* Field[1].Thickness = 0.1; */
/* Background Field = 1; */

Physical Line("1") = {4};
//+
Physical Line("2") = {3};
//+
Physical Line("3") = {2};
//+
Physical Line("4") = {1};
//+
Line Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
Physical Surface("1") = {1};


