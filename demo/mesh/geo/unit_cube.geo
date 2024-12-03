Mesh.MshFileVersion = 2.0;

/*
                           (x ,y, z)
                    __________
                   /|        /
                  / |       /|
                 /  |______/_|
                /___/_____/  /
                |  /     |  /
                | /      | /
                |/_______|/
            (0, 0, 0)
*/

h = 0.1; //Diameter

Point(1) = {0, 0, 0, h};
Point(2) = {1, 0, 0, h};
Point(3) = {0, 1, 0, h};
Point(4) = {0, 0, 1, h};
Point(5) = {1, 1, 0, h};
Point(6) = {0, 1, 1, h};
Point(7) = {1, 0, 1, h};
Point(8) = {1, 1, 1, h};

Line(1) = {7, 2};
Line(2) = {1, 4};
Line(3) = {7, 4};
Line(4) = {2, 1};
Line(5) = {5, 2};
Line(6) = {5, 8};
Line(7) = {7, 8};
Line(8) = {6, 8};
Line(9) = {4, 6};
Line(10) = {1, 3};
Line(11) = {6, 3};
Line(12) = {5, 3};

Curve Loop(1) = {2, 9, 11, -10};
Plane Surface(1) = {1};

Curve Loop(2) = {2, -3, 1, 4};
Plane Surface(2) = {2};

Curve Loop(3) = {1, -5, 6, -7};
Plane Surface(3) = {3};

Curve Loop(4) = {6, -8, 11, -12};
Plane Surface(4) = {4};

Curve Loop(5) = {9, 8, -7, 3};
Plane Surface(5) = {5};

Curve Loop(6) = {12, -10, -4, -5};
Plane Surface(6) = {6};

Surface Loop(1) = {2, 1, 5, 4, 3, 6};

Volume(1) = {1};

Physical Surface(1) = {1};
Physical Surface(2) = {2};
Physical Surface(6) = {6};
Physical Surface(3) = {3};
Physical Surface(4) = {4};
Physical Surface(5) = {5};

Physical Volume(19) = {1};
