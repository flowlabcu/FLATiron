Mesh.MshFileVersion = 2.0;
h = 0.1;

// Inlet (centered at y = 0)
Point(1) = {0, -0.5, 0, h};   // Bottom of inlet
Point(2) = {0,  0.5, 0, h};   // Top of inlet
Point(3) = {10, -0.5, 0, h};  // Bottom junction
Point(4) = {10,  0.5, 0, h};  // Top junction

// Bottom branch (symmetric with top)
Point(5) = {20, -4.5, 0, h};  // Bottom tip
Point(6) = {20, -4.0, 0, h};  // Bottom outlet

// Top branch
Point(7) = {20,  4.0, 0, h};  // Top outlet
Point(8) = {20,  4.5, 0, h};  // Top tip

Point(9) = {10, 0.0, 0.0, h};
// Curve definition

//+
Line(1) = {2, 1};
//+
Line(2) = {1, 3};
//+
Line(3) = {3, 5};
//+
Line(4) = {5, 6};
//+
Line(5) = {6, 9};
//+
Line(6) = {9, 7};
//+
Line(7) = {7, 8};
//+
Line(8) = {8, 4};
//+
Line(9) = {4, 2};
//+
Curve Loop(1) = {9, 1, 2, 3, 4, 5, 6, 7, 8};
//+
Plane Surface(1) = {1};
//+
Physical Curve("in", 10) = {1};
//+
Physical Curve("out1", 11) = {4};
//+
Physical Curve("out2", 12) = {7};
//+
Physical Curve("walls", 13) = {2, 9, 8, 6, 5, 3};
//+
Physical Surface("domain", 14) = {1};
