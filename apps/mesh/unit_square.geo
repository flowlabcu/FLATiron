h = 0.01;
//+
Point(1) = {0, 0, 0, h};
//+
Point(2) = {0, 1, 0, h};
//+
Point(3) = {1, 0, 0, h};
//+
Point(4) = {1, 1, 0, h};
//+
Line(1) = {2, 1};
//+
Line(2) = {1, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 2};
//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Plane Surface(1) = {1};
//+
Physical Curve(1) = {1};
//+
Physical Curve(2) = {2};
//+
Physical Curve(3) = {3};
//+
Physical Curve(4) = {4};
//+

//+
Physical Surface(5) = {1};
