Point(1) = {1.5, 0, 0, 1.0};
Point(2) = {1.4, -0.1, 0, 1.0};
Point(3) = {1.7, -0.2, 0, 1.0};
Point(4) = {1.5, -0.1, 0, 1.0};
Line(1) = {1, 2};
Line(2) = {3, 2};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(5) = {4, 1, -2, 3};
Plane Surface(6) = {5};
Physical Line(7) = {1};
Physical Line(8) = {2};
Physical Line(9) = {3};
Physical Line(10) = {4};
Physical Surface(11) = {6};
