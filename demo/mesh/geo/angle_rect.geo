Mesh.MshFileVersion = 2.0;

/*
1-------------------2
|   |   |   |   |   |
|   |   |   |   |   |
4-------------------3
*/

// -- Parameters
dx = 0.05;

x1 = 0.64644661;
y1 = -0.56066017;
x2 = 2.06066017;
y2 = 0.85355339;
x3 = 1.35355339;
y3 = 1.56066017;
x4 = -0.06066017;
y4 = 0.14644661;


// -- Points
Point(1) = {x1, y1, 0, dx};
Point(2) = {x2, y2, 0, dx};
Point(3) = {x3, y3, 0, dx};
Point(4) = {x4, y4, 0, dx};


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


//+
Plane Surface(1) = {1};
//+
Physical Surface("1") = {1};

