Mesh.MshFileVersion = 2.0;

/*
1-------------------2
|   |   |   |   |   |
|   |   |   |   |   |
4-------------------3
*/

// -- Parameters
hin = 4.0;
D = 0.1;
w  = 22.*D;
h  = 4.1*D;
dx = D/hin/2;

// -- Points
Point(1) = {0, h, 0, dx};
Point(2) = {w, h, 0, dx};
Point(3) = {w, 0, 0, dx};
Point(4) = {0, 0, 0, dx};

Point(5) = {2*D, 2*D, 0, dx};
Point(6) = {2*D+D/2., 2*D, 0, dx};
Point(7) = {2*D-D/2., 2*D, 0, dx};

Circle(5) = {6,5,7};
Circle(6) = {7,5,6};

// -- Lines
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};


Physical Line("1") = {4};
//+
Physical Line("2") = {3};
//+
Physical Line("3") = {2};
//+
Physical Line("4") = {1};
//+
Line Loop(1) = {1, 2, 3, 4};
Line Loop(2) = {5,6};
Physical Curve("5") = {5, 6};
//+



//+
Plane Surface(1) = {1, 2};
//+
Physical Surface("1") = {1};

Field[1] = Box;
Field[1].VIn = dx/2.;
Field[1].VOut = dx;
Field[1].XMax = 2*D+5*D;
Field[1].XMin = 2*D-1.5*D;
Field[1].YMax = 2*D+1.5*D;
Field[1].YMin = 2*D-1.5*D;
Field[1].ZMax = 0;
Field[1].ZMin = 0;
Background Field = 1;


