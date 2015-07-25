// Gmsh project created on Thu Jul 23 19:53:32 2015
Point(1) = {0, 0, 0, 1.0};
Point(2) = {200, 0, 0, 1.0};
Point(3) = {200, 70, 0, 1.0};
Point(4) = {147.9, 70, 0, 1.0};
Point(5) = {77.287, 169.06, 0, 1.0};
Point(6) = {73.477, 176.68, 0, 1.0};
Point(7) = {73.477, 191.92, 0, 1.0};
Point(8) = {63.727, 191.92, 0, 1.0};
Point(9) = {63.727, 169.06, 0, 1.0};
Point(10) = {52.1, 70, 0, 1.0};
Point(11) = {0, 70, 0, 1.0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 9};
Line(9) = {9, 10};
Line(10) = {10, 11};
Line(11) = {11, 1};
Line Loop(12) = {9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8};
Plane Surface(13) = {12};
Physical Line(14) = {1};
Physical Line(15) = {2};
Physical Line(16) = {3};
Physical Line(17) = {4};
Physical Line(18) = {5};
Physical Line(19) = {6};
Physical Line(20) = {7};
Physical Line(21) = {8};
Physical Line(22) = {9};
Physical Line(23) = {10};
Physical Line(24) = {11};
Physical Surface(25) = {13};
