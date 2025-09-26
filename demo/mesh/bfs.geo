h=0.1;
//+
Point(1) = {0, 0, 0, h};
//+
Point(2) = {4, 0, 0, h};
//+
Point(3) = {4, -1, 0, h};
//+
Point(4) = {10, -1, 0, h};
//+
Point(5) = {10, 1, 0, h};
//+
Point(6) = {0, 1, 0, h};
//+
Line(1) = {6, 1};
//+
Line(2) = {1, 2};
//+
Line(3) = {2, 3};
//+
Line(4) = {3, 4};
//+
Line(5) = {4, 5};
//+
Line(6) = {5, 6};
//+
Curve Loop(1) = {6, 1, 2, 3, 4, 5};
//+
Plane Surface(1) = {1};
//+
Physical Curve(7) = {1};
//+
Physical Curve(8) = {2, 3, 4};
//+
Physical Curve(9) = {5};
//+
Physical Curve(10) = {6};
//+
Physical Surface(11) = {1};
