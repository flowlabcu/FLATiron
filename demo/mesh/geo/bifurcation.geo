// Gmsh project created on Wed Sep 25 15:33:52 2024
Mesh.MshFileVersion = 2.0;
SetFactory("OpenCASCADE");
h = 0.05;
L = 10.0;
// -- Points
Point(1) = {0.0 ,  0.5, 0.0, h};
Point(2) = {0.0 , -0.5, 0.0, h};
Point(3) = {L,  0.5, 0.0, h};
Point(4) = {L, -0.5, 0.0, h};
Point(5) = {L, 0.0, 0.0, h};

Point(6) = {2*L,  1.5, 0.0, h};
Point(7) = {2*L, -1.5, 0.0, h};
Point(8) = {2*L,  2.0, 0.0, h};
Point(9) = {2*L, -2.0, 0.0, h};

// -- Lines
Line(1) = {1, 2};
Line(2) = {2, 4};
Line(4) = {4, 9};
Line(5) = {9, 7};
Line(6) = {7, 5};
Line(7) = {5, 6};
Line(8) = {6, 8};
Line(9) = {8, 3};
Line(10) = {3, 1};

//Line(2) = {2, 3};
//Line(3) = {3, 4};
//Line(4) = {4, 1};//+
Physical Curve("in", 11) = {1};
//+
Physical Curve("out1", 12) = {5};
//+
Physical Curve("out2", 13) = {8};
//+
Physical Curve("walls", 14) = {2, 4, 6, 7, 9, 10};
//+
Curve Loop(1) = {10, 1, 2, 4, 5, 6, 7, 8, 9};
//+
Plane Surface(1) = {1};
//+
Physical Surface("domain", 15) = {1};
