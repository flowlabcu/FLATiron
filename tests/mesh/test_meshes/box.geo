Mesh.MshFileVersion = 2.0;
SetFactory("OpenCASCADE");

h = 0.05;

Point(1) = {0, 0, 0, h};
Point(2) = {2, 0, 0, h};
Point(3) = {2, 1, 0, h};
Point(4) = {0, 1, 0, h};
//
// Adding the y = 1 line
//
Line(1) = {4, 3};
//
// Adding the x = 2 line
//
Line(2) = {3, 2};
//
// Adding the y = 0 line
//
Line(3) = {2, 1};
//
// Adding the x = 0 line
//
Line(4) = {1, 4};
//
// creating the surface
//
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
//
// labelling the x = 0 line as 5
//
Physical Curve(5) = {4};
//
// labelling the y = 1 line as 6
//
Physical Curve(6) = {1};
//
// labelling the x = 2 line as 7
//
Physical Curve(7) = {2};
//
// labelling the y = 0 line as 8
//
Physical Curve(8) = {3};
//
// the surface itself is labelled with ID 9
//
Physical Surface(9) = {1};
