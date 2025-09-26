Mesh.MshFileVersion = 2.0;
h = 0.1;
//+
Point(1) = {13, 0, 0, h};
//+
Point(2) = {18, -10, 0, h};
//+
Point(3) = {13,  -10, 0, h};
//+
Point(4) = {18, 0, 0, h};//+
Line(1) = {1, 3};
//+
Line(2) = {3, 2};
//+
Line(3) = {2, 4};
//+
Line(4) = {4, 1};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
//+
Physical Curve(5) = {1, 4, 3, 2};
//+
Physical Surface(6) = {1};
