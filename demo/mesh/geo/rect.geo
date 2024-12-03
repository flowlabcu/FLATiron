Mesh.MshFileVersion = 2.0;

/*

1-------------------2
|   |   |   |   |   |
|   |   |   |   |   |
4-------------------3

*/

// -- Parameters
w  = 1;
h  = 0.1;
dx = w/500; 

// -- Points
Point(1) = {0, h, 0, dx};
Point(2) = {w, h, 0, dx};
Point(3) = {w, 0, 0, dx};
Point(4) = {0, 0, 0, dx};

// -- Lines
Line(4) = {1, 2};
Line(3) = {2, 3};
Line(2) = {3, 4};
Line(1) = {4, 1};

// -- Add physical ids for the lines and surface
Physical Line("1") = {1};
Physical Line("2") = {2};
Physical Line("3") = {3};
Physical Line("4") = {4};
Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Physical Surface("1") = {1};


